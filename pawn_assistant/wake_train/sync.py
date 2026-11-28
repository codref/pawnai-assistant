"""Share wake-word clips and models through pawn-server.

* ``pull``: clips the phone (or any assistant) uploaded → ``<project>/inbox/<label>/``,
  with the usual JSON sidecar plus ``source: server``. Already pulled clip ids
  are remembered in ``inbox/.server_pulled.json``.
* ``push``: this laptop's own harvested inbox clips → the server (sidecars get
  ``uploaded: true``), so every assistant's clips live in one place.
* ``publish``: a trained ``.onnx`` → the server's active model.
* ``ensure_model``: download the active model when its sha256 differs from the
  local copy (``wake.auto_update`` in ``serve``).

pawn-server side: pawnai ``docs/ASSISTANTS.md`` (``/v1/assistants/wake/*``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pawn_assistant.wake_train.harvest import LABELS, inbox_clips

logger = logging.getLogger(__name__)

PULLED_FILE = ".server_pulled.json"


@dataclass
class SyncReport:
    added: int = 0
    skipped: int = 0
    failed: int = 0


def _pulled(root: Path) -> set[str]:
    path = Path(root) / "inbox" / PULLED_FILE
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return set()


def _save_pulled(root: Path, ids: set[str]) -> None:
    path = Path(root) / "inbox" / PULLED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(ids)), encoding="utf-8")


def _stamp(iso: Optional[str]) -> str:
    text = (iso or "").replace("-", "").replace(":", "").replace("T", "-")
    return text[:15] if len(text) >= 15 else "00000000-000000"


def pull_clips(admin: Any, root: Path, *, labels: Optional[list[str]] = None) -> SyncReport:
    """Download server clips not pulled before into the project's inbox."""
    root = Path(root)
    seen = _pulled(root)
    report = SyncReport()
    for clip in admin.list_clips():
        cid = str(clip.get("id") or "")
        label = str(clip.get("label") or "")
        if not cid or label not in LABELS or (labels and label not in labels):
            report.skipped += 1
            continue
        if cid in seen:
            report.skipped += 1
            continue
        try:
            data = admin.download_clip(cid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("wake pull: %s failed: %s", cid, exc)
            report.failed += 1
            continue
        score = float(clip.get("score") or 0.0)
        folder = root / "inbox" / label
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{_stamp(clip.get('captured_at') or clip.get('created_at'))}_{score:.2f}_srv{cid[:8]}.wav"
        path.write_bytes(data)
        meta = {
            "label": label,
            "score": round(score, 4),
            "time": _stamp(clip.get("captured_at") or clip.get("created_at")),
            "source": "server",
            "clip_id": cid,
            "device_id": clip.get("device_id"),
            **{k: v for k, v in (clip.get("meta") or {}).items() if k not in ("label", "score")},
        }
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
        seen.add(cid)
        report.added += 1
    _save_pulled(root, seen)
    return report


def push_clips(device: Any, root: Path, *, labels: Optional[list[str]] = None) -> SyncReport:
    """Upload this laptop's harvested inbox clips (not ones pulled from the server)."""
    report = SyncReport()
    for clip in inbox_clips(Path(root), labels):
        if clip.meta.get("source") == "server" or clip.meta.get("uploaded"):
            report.skipped += 1
            continue
        try:
            device.upload_clip(
                clip.path.read_bytes(),
                label=clip.label,
                score=clip.score,
                meta={k: v for k, v in clip.meta.items() if k in ("text", "intent", "reason")},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("wake push: %s failed: %s", clip.path.name, exc)
            report.failed += 1
            continue
        meta = {**clip.meta, "uploaded": True}
        clip.path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
        report.added += 1
    return report


def publish_model(admin: Any, model: Path, *, notes: str = "", version: str = "", thresholds: Optional[dict] = None) -> dict:
    name = Path(model).stem.split(".")[0] or "hey_pawn"
    return admin.publish_model(Path(model).read_bytes(), version=version, notes=notes, thresholds=thresholds, name=name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(device: Any, dest_dir: Path) -> Optional[tuple[Path, dict[str, Any]]]:
    """Return ``(local path, model info)`` for the server's active model, downloading if needed.

    None when the server has no model (or the call fails: keep the configured one).
    """
    try:
        model = device.current_model()
    except Exception as exc:  # noqa: BLE001
        logger.info("wake auto-update: server unavailable (%s)", exc)
        return None
    if not model:
        return None
    dest_dir = Path(dest_dir)
    path = dest_dir / f"{model['name']}.{model['version']}.onnx"
    if path.is_file() and sha256_file(path) == model["sha256"]:
        return path, model
    try:
        data = device.download_model(model["version"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("wake auto-update: download failed: %s", exc)
        return None
    if hashlib.sha256(data).hexdigest() != model["sha256"]:
        logger.warning("wake auto-update: checksum mismatch for %s — ignored", model["version"])
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    logger.info("wake auto-update: downloaded %s", path.name)
    return path, model
