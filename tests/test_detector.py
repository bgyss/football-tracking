from __future__ import annotations

import numpy as np
import pytest

from football_tracking.cache import CacheMismatch, DetectionCache
from football_tracking.detector import Detection, RFDETRDetector


class FakeModel:
    class_names = {0: "player", 1: "official"}

    def __init__(self) -> None:
        self.seen: np.ndarray | None = None

    def predict(self, image: np.ndarray, threshold: float = 0.1) -> dict:
        self.seen = image.copy()
        return {
            "xyxy": np.array([[1, 2, 11, 22]], dtype=np.float32),
            "confidence": np.array([0.8], dtype=np.float32),
            "class_id": np.array([0], dtype=np.int32),
        }


def test_rfdetr_adapter_converts_bgr_to_rgb_and_normalizes_output() -> None:
    model = FakeModel()
    detector = RFDETRDetector(model_size="small", checkpoint=None, model=model)
    frame_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    frame_bgr[0, 0] = [3, 2, 1]

    detections = detector.predict(frame_bgr)

    assert model.seen is not None
    assert model.seen[0, 0].tolist() == [1, 2, 3]
    assert len(detections) == 1
    assert detections[0].box_xyxy == (1.0, 2.0, 11.0, 22.0)
    assert detections[0].score == pytest.approx(0.8)
    assert detections[0].class_name == "player"


def test_detection_rejects_invalid_box() -> None:
    with pytest.raises(ValueError, match="box"):
        Detection((2.0, 2.0, 1.0, 3.0), 0.5, "player", 0)


def test_detection_cache_round_trip_and_config_invalidation(tmp_path) -> None:
    path = tmp_path / "detections.jsonl"
    frames = {0: [Detection((1.0, 2.0, 11.0, 22.0), 0.8, "player", 0)], 1: []}

    DetectionCache.save(path, source_hash="video", detector_config_hash="cfg-1", frames=frames)
    loaded = DetectionCache.load(path, source_hash="video", detector_config_hash="cfg-1")

    assert loaded == frames
    with pytest.raises(CacheMismatch, match="configuration"):
        DetectionCache.load(path, source_hash="video", detector_config_hash="cfg-2")


def test_strict_detection_cache_rejects_duplicate_and_wrong_provenance(tmp_path) -> None:
    path = tmp_path / "shared.jsonl"
    frames = {0: [Detection((0, 0, 2, 2), 0.9, "player", 0)], 1: []}
    provenance = {"checkpoint_sha256": "checkpoint", "class_mapping": {"0": "player"}, "proxy": False}
    DetectionCache.save(path, "source", "detector", frames, provenance=provenance)

    assert DetectionCache.load_strict(path, "source", "detector", frame_count=2, provenance=provenance)[1] == []
    with pytest.raises(CacheMismatch, match="checkpoint"):
        DetectionCache.load_strict(path, "source", "detector", frame_count=2, provenance={**provenance, "checkpoint_sha256": "wrong"})

    path.write_text(path.read_text() + '{"frame_index":1,"detections":[]}\n', encoding="utf-8")
    with pytest.raises(CacheMismatch, match="duplicate"):
        DetectionCache.load_strict(path, "source", "detector", frame_count=2, provenance=provenance)


def test_strict_detection_cache_rejects_missing_frames_and_proxy_provenance(tmp_path) -> None:
    path = tmp_path / "shared.jsonl"
    provenance = {"checkpoint_sha256": "checkpoint", "class_mapping": {"0": "player"}, "proxy": False}
    DetectionCache.save(path, "source", "detector", {0: []}, provenance=provenance)

    with pytest.raises(CacheMismatch, match="source"):
        DetectionCache.load_strict(path, "wrong-source", "detector", frame_count=1, provenance=provenance)
    with pytest.raises(CacheMismatch, match="missing"):
        DetectionCache.load_strict(path, "source", "detector", frame_count=2, provenance=provenance)
    with pytest.raises(CacheMismatch, match="proxy"):
        DetectionCache.load_strict(path, "source", "detector", frame_count=1, provenance={**provenance, "proxy": True})


def test_rfdetr_adapter_prefers_per_detection_class_name_metadata() -> None:
    class ModelWithMetadata(FakeModel):
        def predict(self, image: np.ndarray, threshold: float = 0.1) -> dict:
            return {
                "xyxy": np.array([[1, 2, 11, 22]], dtype=np.float32),
                "confidence": np.array([0.8], dtype=np.float32),
                "class_id": np.array([1], dtype=np.int32),
                "data": {"class_name": np.array(["person"], dtype=object)},
            }

    detector = RFDETRDetector(model_size="small", model=ModelWithMetadata())

    assert detector.predict(np.zeros((4, 4, 3), dtype=np.uint8))[0].class_name == "person"
