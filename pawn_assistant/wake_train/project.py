"""Wake-word training project layout and manifest."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


def slugify(phrase: str) -> str:
    s = phrase.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "wake"


@dataclass
class WakeProject:
    root: Path
    phrase: str
    model_name: str
    sample_rate: int = 16_000
    clip_seconds: float = 2.0
    record_seconds: float = 1.6

    @property
    def positives_dir(self) -> Path:
        return self.root / "positives"

    @property
    def negatives_dir(self) -> Path:
        return self.root / "negatives"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def model_path(self) -> Path:
        return self.root / f"{self.model_name}.onnx"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def ensure(self) -> None:
        for d in (
            self.root,
            self.positives_dir,
            self.negatives_dir,
            self.features_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self.save()

    def save(self) -> None:
        self.manifest_path.write_text(
            json.dumps(asdict(self), indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def create(cls, root: Path, phrase: str) -> "WakeProject":
        name = slugify(phrase)
        proj = cls(root=root.resolve(), phrase=phrase, model_name=name)
        proj.ensure()
        proj.save()
        return proj

    @classmethod
    def load(cls, root: Path) -> "WakeProject":
        root = root.resolve()
        data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            root=root,
            phrase=data["phrase"],
            model_name=data["model_name"],
            sample_rate=int(data.get("sample_rate", 16_000)),
            clip_seconds=float(data.get("clip_seconds", 2.0)),
            record_seconds=float(data.get("record_seconds", 1.6)),
        )

    def counts(self) -> dict[str, Any]:
        pos = list(self.positives_dir.glob("*.wav"))
        neg_by_cat: dict[str, int] = {}
        for p in self.negatives_dir.rglob("*.wav"):
            cat = p.parent.name if p.parent != self.negatives_dir else "misc"
            neg_by_cat[cat] = neg_by_cat.get(cat, 0) + 1
        return {
            "phrase": self.phrase,
            "positives": len(pos),
            "negatives": sum(neg_by_cat.values()),
            "negatives_by_category": neg_by_cat,
            "model_exists": self.model_path.exists(),
            "model_path": str(self.model_path),
        }
