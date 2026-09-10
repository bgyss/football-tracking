"""Per-shot tracking adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from .detector import Detection


@dataclass(frozen=True, slots=True)
class TrackObservation:
    tracklet_id: str
    frame_index: int
    pts: int
    bbox_xyxy_px: tuple[float, float, float, float]
    score: float
    state: str = "observed"

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy_px
        if x2 <= x1 or y2 <= y1:
            raise ValueError("track box must have positive width and height")
        if self.frame_index < 0 or self.pts < 0:
            raise ValueError("frame_index and pts must be non-negative")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("track score must be between 0 and 1")
        if self.state not in {"observed", "predicted"}:
            raise ValueError("unsupported track state")


class TrackerAdapter(Protocol):
    def update(
        self,
        detections: Sequence[Detection],
        frame_bgr: np.ndarray | None,
        timestamp_s: float,
        frame_index: int | None = None,
        pts: int | None = None,
    ) -> list[TrackObservation]: ...

    def reset(self, shot_id: str | None = None) -> None: ...


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


@dataclass(slots=True)
class _IoUTrack:
    identifier: int
    box: np.ndarray
    score: float
    misses: int = 0


class IoUTracker:
    """Deterministic motion-only fallback used for tests and smoke benchmarks."""

    def __init__(self, fps: float, lost_track_buffer: int = 30, iou_threshold: float = 0.2) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if lost_track_buffer < 0:
            raise ValueError("lost_track_buffer must be non-negative")
        self.fps = float(fps)
        self.lost_track_buffer = int(lost_track_buffer)
        self.iou_threshold = float(iou_threshold)
        self._tracks: dict[int, _IoUTrack] = {}
        self._next_identifier = 1
        self._shot_id = "shot-0"

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def reset(self, shot_id: str | None = None) -> None:
        self._tracks.clear()
        self._next_identifier = 1
        if shot_id is not None:
            self._shot_id = str(shot_id)

    def update(
        self,
        detections: Sequence[Detection],
        frame_bgr: np.ndarray | None,
        timestamp_s: float,
        frame_index: int | None = None,
        pts: int | None = None,
    ) -> list[TrackObservation]:
        if timestamp_s < 0:
            raise ValueError("timestamp_s must be non-negative")
        resolved_frame = int(round(timestamp_s * self.fps)) if frame_index is None else int(frame_index)
        resolved_pts = resolved_frame if pts is None else int(pts)
        for track in self._tracks.values():
            track.misses += 1
        candidate_edges: list[tuple[float, int, int]] = []
        boxes = [np.asarray(detection.box_xyxy, dtype=float) for detection in detections]
        for track_id, track in self._tracks.items():
            for detection_index, box in enumerate(boxes):
                score = _iou(track.box, box)
                if score >= self.iou_threshold:
                    candidate_edges.append((score, track_id, detection_index))
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        for _, track_id, detection_index in sorted(candidate_edges, key=lambda edge: (-edge[0], edge[1], edge[2])):
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)
            track = self._tracks[track_id]
            track.box = boxes[detection_index]
            track.score = float(detections[detection_index].score)
            track.misses = 0
        for detection_index, box in enumerate(boxes):
            if detection_index in assigned_detections:
                continue
            track_id = self._next_identifier
            self._next_identifier += 1
            self._tracks[track_id] = _IoUTrack(track_id, box, float(detections[detection_index].score))
            assigned_tracks.add(track_id)
        expired = [track_id for track_id, track in self._tracks.items() if track.misses > self.lost_track_buffer]
        for track_id in expired:
            del self._tracks[track_id]
        result: list[TrackObservation] = []
        for track_id in sorted(assigned_tracks):
            track = self._tracks.get(track_id)
            if track is None:
                continue
            result.append(
                TrackObservation(
                    f"{self._shot_id}:t{track_id}",
                    resolved_frame,
                    resolved_pts,
                    tuple(float(value) for value in track.box),
                    track.score,
                )
            )
        return result


def _as_detections(detections: Sequence[Detection]) -> Any:
    import supervision as sv  # type: ignore[import-not-found]

    if not detections:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.asarray([detection.box_xyxy for detection in detections], dtype=float),
        confidence=np.asarray([detection.score for detection in detections], dtype=float),
        class_id=np.asarray([detection.class_id for detection in detections], dtype=int),
    )


def _backend_rows(output: Any) -> list[tuple[int, tuple[float, float, float, float], float]]:
    if output is None:
        return []
    if isinstance(output, (list, tuple)):
        rows: list[tuple[int, tuple[float, float, float, float], float]] = []
        for item in output:
            if isinstance(item, dict):
                identifier = item.get("tracker_id", item.get("track_id"))
                box = item.get("xyxy", item.get("bbox_xyxy_px"))
                score = item.get("confidence", item.get("score", 1.0))
            else:
                identifier = getattr(item, "tracker_id", getattr(item, "track_id", None))
                box = getattr(item, "xyxy", getattr(item, "bbox_xyxy_px", None))
                score = getattr(item, "confidence", getattr(item, "score", 1.0))
            if identifier is not None and box is not None:
                rows.append((int(identifier), tuple(float(value) for value in box), float(score)))
        return rows
    tracker_ids = getattr(output, "tracker_id", None)
    boxes = getattr(output, "xyxy", None)
    scores = getattr(output, "confidence", None)
    if tracker_ids is None or boxes is None:
        return []
    tracker_ids = np.asarray(tracker_ids).reshape(-1)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.ones(len(boxes), dtype=float) if scores is None else np.asarray(scores, dtype=float).reshape(-1)
    return [
        (int(identifier), tuple(float(value) for value in box), float(score))
        for identifier, box, score in zip(tracker_ids, boxes, scores)
        if int(identifier) >= 0
    ]


class RoboflowTracker:
    """Lazy adapter for Roboflow's maintained ByteTrack/BoT-SORT package."""

    def __init__(
        self,
        kind: str = "botsort",
        fps: float = 30.0,
        lost_track_buffer: int = 30,
        enable_cmc: bool = True,
        backend: Any | None = None,
        shot_id: str = "shot-0",
    ) -> None:
        normalized = kind.lower().replace("-", "")
        if normalized not in {"botsort", "bytetrack"}:
            raise ValueError("kind must be 'botsort' or 'bytetrack'")
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.kind = normalized
        self.fps = float(fps)
        self.lost_track_buffer = int(lost_track_buffer)
        self.enable_cmc = bool(enable_cmc)
        self._external_backend = backend is not None
        self.backend = backend if backend is not None else self._load_backend()
        self._shot_id = shot_id
        self._active_ids: set[int] = set()

    def _load_backend(self) -> Any:
        try:
            import trackers  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("Roboflow trackers is unavailable; install the roboflow optional dependencies") from error
        tracker_type = trackers.BoTSORTTracker if self.kind == "botsort" else trackers.ByteTrackTracker
        kwargs = {"frame_rate": self.fps, "lost_track_buffer": self.lost_track_buffer}
        if self.kind == "botsort":
            kwargs["enable_cmc"] = self.enable_cmc
        try:
            return tracker_type(**kwargs)
        except TypeError:
            kwargs.pop("frame_rate", None)
            try:
                return tracker_type(**kwargs)
            except TypeError:
                return tracker_type()

    def reset(self, shot_id: str | None = None) -> None:
        self._active_ids.clear()
        if shot_id is not None:
            self._shot_id = str(shot_id)
        reset = getattr(self.backend, "reset", None)
        if callable(reset):
            reset()

    def update(
        self,
        detections: Sequence[Detection],
        frame_bgr: np.ndarray | None,
        timestamp_s: float,
        frame_index: int | None = None,
        pts: int | None = None,
    ) -> list[TrackObservation]:
        resolved_frame = int(round(timestamp_s * self.fps)) if frame_index is None else int(frame_index)
        resolved_pts = resolved_frame if pts is None else int(pts)
        backend_input = detections if self._external_backend else _as_detections(detections)
        backend_frame = frame_bgr if self.kind == "botsort" else None
        try:
            output = self.backend.update(backend_input, frame=backend_frame, timestamp=timestamp_s)
        except TypeError:
            output = self.backend.update(backend_input, frame=backend_frame)
        rows = _backend_rows(output)
        self._active_ids = {identifier for identifier, _, _ in rows}
        return [
            TrackObservation(f"{self._shot_id}:t{identifier}", resolved_frame, resolved_pts, box, score)
            for identifier, box, score in sorted(rows, key=lambda row: row[0])
        ]

    @property
    def active_track_count(self) -> int:
        return len(self._active_ids)
