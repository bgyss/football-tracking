"""Detector protocol and lazy RF-DETR adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    class_name: str
    class_id: int

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.box_xyxy
        if any(not np.isfinite(float(value)) for value in self.box_xyxy):
            raise ValueError("box coordinates must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box must have positive width and height")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "box_xyxy": list(self.box_xyxy),
            "score": float(self.score),
            "class_name": self.class_name,
            "class_id": int(self.class_id),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Detection":
        return cls(tuple(float(item) for item in value["box_xyxy"]), float(value["score"]), str(value["class_name"]), int(value["class_id"]))


class Detector(Protocol):
    def predict(self, frame_bgr: np.ndarray) -> list[Detection]: ...


def _class_name(class_names: Any, class_id: int) -> str:
    if isinstance(class_names, dict):
        return str(class_names.get(class_id, class_names.get(str(class_id), f"class_{class_id}")))
    if isinstance(class_names, (list, tuple)) and class_id < len(class_names):
        return str(class_names[class_id])
    return "player" if class_id == 0 else f"class_{class_id}"


def _field(output: Any, name: str, default: Any = None) -> Any:
    if isinstance(output, dict):
        return output.get(name, default)
    return getattr(output, name, default)


class RFDETRDetector:
    """Adapt RF-DETR predictions to the project's small Detection contract."""

    def __init__(
        self,
        model_size: str = "small",
        checkpoint: str | None = None,
        device: str | None = None,
        threshold: float = 0.1,
        model: Any | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        model_size = model_size.lower()
        if model_size not in {"small", "medium"}:
            raise ValueError("model_size must be 'small' or 'medium'")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.model_size = model_size
        self.checkpoint = checkpoint
        self.device = device
        self.threshold = threshold
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None and model is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["RF_HOME"] = str(self.cache_dir)
        self.model = model if model is not None else self._load_model()
        self.class_names = getattr(self.model, "class_names", None)

    def _load_model(self) -> Any:
        try:
            import rfdetr  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("RF-DETR is unavailable; install the roboflow optional dependencies") from error
        model_type = rfdetr.RFDETRSmall if self.model_size == "small" else rfdetr.RFDETRMedium
        kwargs: dict[str, Any] = {}
        if self.checkpoint:
            kwargs["pretrain_weights"] = self.checkpoint
        if self.device:
            kwargs["device"] = self.device
        try:
            return model_type(**kwargs)
        except TypeError:
            kwargs.pop("device", None)
            return model_type(**kwargs)

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[-1] != 3:
            raise ValueError("frame_bgr must have shape HxWx3")
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        output = self.model.predict(frame_rgb, threshold=self.threshold)
        boxes = np.asarray(_field(output, "xyxy", []), dtype=float).reshape(-1, 4)
        scores = np.asarray(_field(output, "confidence", _field(output, "scores", [])), dtype=float).reshape(-1)
        class_ids = np.asarray(_field(output, "class_id", _field(output, "class_ids", [])), dtype=int).reshape(-1)
        if len(scores) != len(boxes) or len(class_ids) != len(boxes):
            raise ValueError("RF-DETR output arrays have inconsistent lengths")
        class_names = _field(output, "class_names", self.class_names)
        data = _field(output, "data", {})
        per_detection_names = None
        if isinstance(data, dict) and "class_name" in data:
            per_detection_names = np.asarray(data["class_name"], dtype=object).reshape(-1)
            if len(per_detection_names) != len(boxes):
                raise ValueError("RF-DETR class_name metadata has inconsistent length")
        return [
            Detection(tuple(float(value) for value in box), float(score), str(per_detection_names[index]) if per_detection_names is not None else _class_name(class_names, int(class_id)), int(class_id))
            for index, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids))
            if 0.0 <= float(score) <= 1.0
        ]


class SyntheticDetector:
    """Deterministic proxy detector for plumbing and throughput benchmarks only."""

    is_proxy = True

    def __init__(self, boxes: Sequence[tuple[float, float, float, float]] | None = None) -> None:
        self.boxes = boxes

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        boxes = self.boxes or [
            (0.10 * width, 0.20 * height, 0.18 * width, 0.78 * height),
            (0.62 * width, 0.18 * height, 0.70 * width, 0.76 * height),
        ]
        return [Detection(tuple(float(value) for value in box), 0.99, "player", 0) for box in boxes]
