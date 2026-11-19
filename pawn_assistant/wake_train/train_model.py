"""Train a small DNN and export an openWakeWord-compatible ONNX classifier."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from pawn_assistant.wake_train.features import N_FEATURES, N_FRAMES, build_feature_arrays
from pawn_assistant.wake_train.project import WakeProject

logger = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "torch is required for wake-train. Install with:\n"
            "  uv sync --extra wake-train\n"
            "  # or: pip install 'pawn-assistant[wake-train]'"
        ) from exc
    return torch, nn


def _build_net(nn, layer_dim: int = 32):
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flatten = nn.Flatten()
            self.layer1 = nn.Linear(N_FRAMES * N_FEATURES, layer_dim)
            self.relu1 = nn.ReLU()
            self.norm1 = nn.LayerNorm(layer_dim)
            self.layer2 = nn.Linear(layer_dim, layer_dim)
            self.relu2 = nn.ReLU()
            self.norm2 = nn.LayerNorm(layer_dim)
            self.out = nn.Linear(layer_dim, 1)
            self.act = nn.Sigmoid()

        def forward(self, x):  # noqa: ANN001
            x = self.relu1(self.norm1(self.layer1(self.flatten(x))))
            x = self.relu2(self.norm2(self.layer2(x)))
            return self.act(self.out(x))

    return Net()


def train_and_export(
    project: WakeProject,
    *,
    steps: int = 4000,
    batch_size: int = 64,
    layer_dim: int = 32,
    lr: float = 1e-3,
    augment_positives: int = 4,
    seed: int = 0,
) -> Path:
    torch, nn = _require_torch()
    # Ensure OWW feature ONNX models exist.
    try:
        from openwakeword import utils as oww_utils

        oww_utils.download_models()
    except Exception as exc:  # noqa: BLE001
        logger.warning("download_models: %s", exc)

    train_x, train_y, val_x, val_y = build_feature_arrays(
        project, augment_positives=augment_positives, seed=seed
    )

    device = torch.device("cpu")
    model = _build_net(nn, layer_dim=layer_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    tx = torch.from_numpy(train_x)
    ty = torch.from_numpy(train_y).unsqueeze(1)
    vx = torch.from_numpy(val_x)
    vy = torch.from_numpy(val_y).unsqueeze(1)

    best_state = None
    best_score = -1e9
    rng = np.random.default_rng(seed)
    # Class-balanced batches: negatives (esp. speech) far outnumber positives.
    pos_idx = np.flatnonzero(train_y >= 0.5)
    neg_idx = np.flatnonzero(train_y < 0.5)
    half = max(1, batch_size // 2)

    model.train()
    for step in range(1, steps + 1):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=half, replace=pos_idx.size < half),
                rng.choice(neg_idx, size=half, replace=neg_idx.size < half),
            ]
        )
        xb = tx[idx].to(device)
        yb = ty[idx].to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vp = model(vx.to(device)).cpu().numpy().reshape(-1)
                vt = val_y.reshape(-1)
                thr = 0.5
                pred_pos = vp >= thr
                truth_pos = vt >= 0.5
                tp = float(np.logical_and(pred_pos, truth_pos).sum())
                fp = float(np.logical_and(pred_pos, ~truth_pos).sum())
                fn = float(np.logical_and(~pred_pos, truth_pos).sum())
                recall = tp / max(1.0, tp + fn)
                precision = tp / max(1.0, tp + fp)
                # Silence-ish negatives are the last synthetic block — approximate
                # via low-energy: use all negatives' max score.
                neg_scores = vp[vt < 0.5]
                pos_scores = vp[vt >= 0.5]
                # False wakes are the costly error for an always-on mic.
                score = recall - 2.0 * (fp / max(1.0, (~truth_pos).sum()))
                logger.info(
                    "step %d loss=%.4f recall=%.2f prec=%.2f "
                    "pos_mean=%.3f neg_mean=%.3f neg_max=%.3f",
                    step,
                    float(loss.item()),
                    recall,
                    precision,
                    float(pos_scores.mean()) if pos_scores.size else 0.0,
                    float(neg_scores.mean()) if neg_scores.size else 0.0,
                    float(neg_scores.max()) if neg_scores.size else 0.0,
                )
                if score >= best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    out = project.model_path
    dummy = torch.rand(1, N_FRAMES, N_FEATURES)
    # Torch 2.4+ defaults toward the dynamo exporter (needs onnxscript).
    # Legacy exporter is enough for this tiny FCN and matches openWakeWord.
    export_kwargs = dict(
        input_names=["input"],
        output_names=[project.model_name],
        opset_version=17,
    )
    try:
        torch.onnx.export(
            model.cpu(),
            dummy,
            str(out),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        # Older torch without the dynamo= flag.
        torch.onnx.export(model.cpu(), dummy, str(out), **export_kwargs)
    logger.info("wrote %s", out)

    # Quick summary metrics at several thresholds
    with torch.no_grad():
        scores = model(vx).numpy().reshape(-1)
    _print_threshold_table(scores, val_y, logger)
    return out


def _print_threshold_table(scores: np.ndarray, labels: np.ndarray, log: logging.Logger) -> None:
    labels = labels.reshape(-1)
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        pred = scores >= thr
        truth = labels >= 0.5
        tp = np.logical_and(pred, truth).sum()
        fp = np.logical_and(pred, ~truth).sum()
        fn = np.logical_and(~pred, truth).sum()
        recall = tp / max(1, tp + fn)
        fpr = fp / max(1, (~truth).sum())
        log.info(
            "threshold %.1f  recall=%.0f%%  false_pos_rate=%.1f%%",
            thr,
            100 * recall,
            100 * fpr,
        )


def live_test(
    model_path: Path,
    *,
    device: Optional[int],
    seconds: float = 20.0,
    threshold: float = 0.5,
) -> None:
    """Stream mic scores for a trained ONNX model."""
    import time

    from openwakeword.model import Model
    from rich.live import Live
    from rich.text import Text

    from pawn_assistant.audio.capture import AudioCapture

    oww = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
    key = list(oww.models.keys())[0]
    print(
        f"Live test {model_path.name} (key={key}) — speak the phrase; Ctrl-C to stop"
    )

    cap = AudioCapture(device=device)
    cap.start()
    t0 = time.monotonic()
    try:
        with Live(refresh_per_second=15) as live:
            while time.monotonic() - t0 < seconds:
                frame = cap.read(timeout=0.5)
                if frame is None:
                    continue
                pcm = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16)
                preds = oww.predict(pcm)
                score = float(preds.get(key, 0.0))
                bar_n = int(max(0, min(30, round(score * 30))))
                mark = " ★" if score >= threshold else ""
                live.update(
                    Text(
                        f"score {score:0.3f}  [{'█' * bar_n}{'·' * (30 - bar_n)}]{mark}",
                        style="green" if score >= threshold else "cyan",
                    )
                )
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap.stop()
