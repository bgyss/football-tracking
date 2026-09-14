from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from football_tracking.detector import Detection
from football_tracking.tracking import IoUTracker, McByteTracker, RoboflowTracker


def detections(*boxes: tuple[float, float, float, float]) -> list[Detection]:
    return [Detection(box, 0.9, "player", 0) for box in boxes]


def test_iou_tracker_keeps_unique_ids_and_expires_after_empty_gap() -> None:
    tracker = IoUTracker(fps=10.0, lost_track_buffer=2)
    first = tracker.update(detections((0, 0, 10, 10), (30, 0, 40, 10)), None, 0.0)
    second = tracker.update(detections((1, 0, 11, 10), (29, 0, 39, 10)), None, 0.1)

    assert len({row.tracklet_id for row in first}) == 2
    assert {row.tracklet_id for row in second} == {row.tracklet_id for row in first}
    assert tracker.update([], None, 0.2) == []
    assert tracker.active_track_count == 2
    tracker.update([], None, 0.3)
    tracker.update([], None, 0.4)
    assert tracker.active_track_count == 0


def test_iou_tracker_reset_starts_a_new_shot_namespace() -> None:
    tracker = IoUTracker(fps=30.0, lost_track_buffer=30)
    first = tracker.update(detections((0, 0, 10, 10)), None, 0.0)
    tracker.reset("shot-1")
    second = tracker.update(detections((0, 0, 10, 10)), None, 0.0)

    assert first[0].tracklet_id == "shot-0:t1"
    assert second[0].tracklet_id == "shot-1:t1"


class TimestampBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[float, bool, int]] = []

    def update(self, detections, frame=None, timestamp=None):
        self.calls.append((timestamp, frame is not None, len(detections)))
        return [{"tracker_id": 7, "xyxy": detections[0].box_xyxy, "confidence": detections[0].score}] if detections else []


def test_roboflow_adapter_passes_bgr_frame_and_timestamps_consistently() -> None:
    backend = TimestampBackend()
    adapter = RoboflowTracker(kind="botsort", fps=59.94, backend=backend)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    result = adapter.update(detections((1, 2, 6, 7)), frame, 1.25)

    assert result[0].tracklet_id == "shot-0:t7"
    assert backend.calls == [(1.25, True, 1)]


class McByteBackend:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.calls: list[tuple[float | None, int]] = []
        self.reset_calls = 0
        self.mask_manager = type("MaskManager", (), {"enabled": True})()

    def update(self, detections, frame=None, timestamp=None):
        assert frame is not None
        self.frames.append(frame)
        self.calls.append((timestamp, len(detections)))
        return [{"tracker_id": 3, "xyxy": detections[0].box_xyxy, "confidence": detections[0].score}] if detections else []

    def reset(self):
        self.reset_calls += 1


def test_mcbyte_adapter_converts_bgr_to_contiguous_rgb_and_preserves_source_coordinates() -> None:
    backend = McByteBackend()
    adapter = McByteTracker(fps=60000 / 1001, device="cpu", enable_masks=True, backend=backend)
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    frame[0, 0] = (3, 2, 1)

    result = adapter.update(detections((1, 2, 3, 4)), frame, 1.25, frame_index=712, pts=89089)

    assert result[0].tracklet_id == "shot-0:t3"
    assert result[0].frame_index == 712
    assert result[0].pts == 89089
    assert backend.frames[0][0, 0].tolist() == [1, 2, 3]
    assert backend.frames[0].flags.c_contiguous
    assert backend.calls == [(1.25, 1)]
    assert adapter.effective_mode()["masks_active"] is True


def test_mcbyte_adapter_advances_empty_frames_and_resets_temporal_state() -> None:
    backend = McByteBackend()
    adapter = McByteTracker(fps=60000 / 1001, device="cpu", enable_masks=True, backend=backend)
    frame = np.zeros((3, 4, 3), dtype=np.uint8)

    assert adapter.update([], frame, 0.0, frame_index=0, pts=0) == []
    adapter.reset("shot-1")

    assert backend.calls == [(0.0, 0)]
    assert backend.reset_calls == 1
    assert adapter.effective_mode()["shot_id"] == "shot-1"


def test_mcbyte_rejects_an_unavailable_requested_mask_device(tmp_path, monkeypatch) -> None:
    unavailable_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", unavailable_torch)
    sam = tmp_path / "sam.pth"
    cutie = tmp_path / "cutie.pth"
    sam.touch()
    cutie.touch()

    with pytest.raises(RuntimeError, match="device 'cuda' is unavailable"):
        McByteTracker(fps=30.0, device="cuda", enable_masks=True, sam_checkpoint=sam, cutie_checkpoint=cutie)
