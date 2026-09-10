from __future__ import annotations

import numpy as np

from football_tracking.detector import Detection
from football_tracking.tracking import IoUTracker, RoboflowTracker


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
