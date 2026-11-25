"""Live wake harvesting, inbox filing, augmentation and threshold choice."""

import json
import time

import numpy as np

from pawn_assistant.wake_train import features as F
from pawn_assistant.wake_train.calibrate import choose_threshold, rates
from pawn_assistant.wake_train.harvest import (
    WakeHarvester,
    file_clip,
    find_project_root,
    inbox_clips,
    inbox_counts,
    resolve_harvest_root,
)


def _project(tmp_path):
    root = tmp_path / "hey_pawn"
    (root / "positives").mkdir(parents=True)
    (root / "negatives").mkdir()
    (root / "manifest.json").write_text(json.dumps({"phrase": "hey pawn", "model_name": "hey_pawn"}))
    model = root / "hey_pawn.onnx"
    model.write_bytes(b"onnx")
    return root, model


def _frames(n, value=0.1):
    return [np.full(1280, value, dtype=np.float32) for _ in range(n)]


def test_auto_harvest_only_inside_a_project(tmp_path):
    root, model = _project(tmp_path)
    assert find_project_root(str(model)) == root.resolve()
    assert resolve_harvest_root("auto", str(model)) == root.resolve()
    assert resolve_harvest_root("auto", "hey_jarvis") is None
    assert resolve_harvest_root(False, str(model)) is None
    assert resolve_harvest_root(str(root), "hey_jarvis") == root.resolve()


def test_wake_clip_is_labelled_by_outcome(tmp_path):
    root, _ = _project(tmp_path)
    h = WakeHarvester(root)
    for f in _frames(40):  # 3.2 s → ring keeps the last 2 s
        h.push(f)
    h.on_wake(0.83)
    h.resolve("confirmed", text="che ore sono", intent="time.now")
    h.resolve("false_wake")  # nothing pending any more: ignored
    clips = inbox_clips(root)
    assert [c.label for c in clips] == ["confirmed"]
    assert clips[0].meta["intent"] == "time.now"
    assert clips[0].score == 0.83
    from pawn_assistant.wake_train.io_wav import read_wav

    audio, sr = read_wav(clips[0].path)
    assert sr == 16_000 and audio.size == 32_000


def test_near_miss_saved_after_peak_unless_wake_follows(tmp_path, monkeypatch):
    root, _ = _project(tmp_path)
    h = WakeHarvester(root, near_miss_gap_s=0)
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    for f in _frames(30):
        h.push(f)
    h.on_score(0.3, 0.25, 0.5)
    h.on_score(0.42, 0.25, 0.5)
    clock[0] += 1.5
    h.push(_frames(1)[0])
    assert inbox_counts(root) == {"near_miss": 1}
    assert inbox_clips(root)[0].score == 0.42

    h.on_score(0.4, 0.25, 0.5)
    h.on_wake(0.7)  # the near miss became a wake: not a near miss
    clock[0] += 1.5
    h.push(_frames(1)[0])
    assert inbox_counts(root) == {"near_miss": 1}


def test_inbox_is_capped(tmp_path):
    root, _ = _project(tmp_path)
    h = WakeHarvester(root, max_per_label=3)
    for f in _frames(30):
        h.push(f)
    for i in range(5):
        h.on_wake(0.6)
        h.resolve("false_wake", n=i)
        time.sleep(0.002)
    assert inbox_counts(root) == {"false_wake": 3}


def test_file_clip_moves_into_training_dirs(tmp_path):
    root, _ = _project(tmp_path)
    h = WakeHarvester(root)
    for f in _frames(30):
        h.push(f)
    for label in ("confirmed", "false_wake", "unsure"):
        h.on_wake(0.7)
        h.resolve(label)
    by = {c.label: c for c in inbox_clips(root)}
    pos = file_clip(root, by["confirmed"], "pos")
    neg = file_clip(root, by["false_wake"], "neg")
    file_clip(root, by["unsure"], "drop")
    assert pos.parent == root / "positives" and pos.name.startswith("live_confirmed_")
    assert neg.parent == root / "negatives" / "live_false_wake"
    assert inbox_clips(root) == []
    assert not list((root / "inbox").rglob("*.json"))


def test_augment_positive_shapes_and_levels():
    rng = np.random.default_rng(1)
    clip = (0.4 * np.sin(np.arange(25_600) / 7)).astype(np.float32)
    bg = [rng.normal(0, 0.1, 16_000).astype(np.float32)]
    for _ in range(30):
        w = F.augment_positive(clip, 32_000, rng, bg)
        assert w.shape == (32_000,)
        assert np.isfinite(w).all() and np.abs(w).max() <= 1.0
    assert F.speed_perturb(clip, 1.1).size == round(clip.size / 1.1)
    noisy = F.mix_at_snr(clip[:16_000], bg[0], 10.0, rng)
    ratio = np.mean(clip[:16_000] ** 2) / np.mean((noisy - clip[:16_000]) ** 2)
    assert 8.0 < 10 * np.log10(ratio) < 12.0


def test_choose_threshold_sits_above_negative_tail():
    pos = [0.95, 0.9, 0.97, 0.88]
    neg = [0.01, 0.05, 0.3, 0.42]
    thr = choose_threshold(pos, neg, margin=0.05)
    assert thr == 0.47
    assert rates(pos, neg, thr) == (1.0, 0)
    # Allowing 25 % false positives ignores the worst negative.
    assert choose_threshold(pos, neg, max_fpr=0.25, margin=0.05) == 0.35


def test_choose_threshold_never_above_median_positive():
    assert choose_threshold([0.6, 0.62, 0.65], [0.99]) == 0.61
