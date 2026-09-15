"""Versioned, human-reviewed annotation manifest validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AnnotationError(ValueError):
    """Raised when an annotation manifest cannot support evaluation."""


@dataclass(frozen=True, slots=True)
class ShotAnnotation:
    shot_id: str
    start_frame: int
    end_frame: int
    play_id: str | None
    split: str

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index < self.end_frame


@dataclass(frozen=True, slots=True)
class AnnotationManifest:
    schema_version: int
    reviewed: bool
    source_sha256: str
    width: int
    height: int
    frame_count: int
    time_base: tuple[int, int]
    shots: dict[str, ShotAnnotation]
    annotations: tuple[dict[str, Any], ...]
    landmarks: tuple[dict[str, Any], ...] = ()


def _box(value: Any) -> tuple[float, float, float, float]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise AnnotationError("bbox_xyxy_px must contain four numeric values") from error
    if len(result) != 4 or result[2] <= result[0] or result[3] <= result[1]:
        raise AnnotationError("bbox_xyxy_px must have positive dimensions")
    return result


def load_annotation_manifest(path: str | Path, source_sha256: str, *, require_reviewed: bool = True) -> AnnotationManifest:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnnotationError(f"unable to read annotation manifest: {error}") from error
    if not isinstance(value, dict):
        raise AnnotationError("annotation manifest must be an object")
    try:
        schema_version = int(value.get("schema_version", 0))
    except (TypeError, ValueError) as error:
        raise AnnotationError("annotation manifest schema_version must be 1") from error
    if schema_version != 1:
        raise AnnotationError("annotation manifest schema_version must be 1")
    reviewed = value.get("reviewed") is True
    if require_reviewed and not reviewed:
        raise AnnotationError("annotation manifest must be explicitly marked reviewed: true")
    source = value.get("source")
    if not isinstance(source, dict) or not source_sha256 or str(source.get("sha256", "")) != source_sha256:
        raise AnnotationError("annotation source sha256 does not match the input video")
    try:
        width, height, frame_count = int(source["width"]), int(source["height"]), int(source["frame_count"])
        numerator, denominator = int(source["time_base"][0]), int(source["time_base"][1])
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise AnnotationError("source metadata is incomplete") from error
    if width <= 0 or height <= 0 or frame_count <= 0 or numerator <= 0 or denominator <= 0:
        raise AnnotationError("source metadata must be positive")
    raw_shots = value.get("shots")
    if not isinstance(raw_shots, dict) or not raw_shots:
        raise AnnotationError("annotation manifest must declare shots")
    shots: dict[str, ShotAnnotation] = {}
    play_splits: dict[str, str] = {}
    for raw_id, raw_shot in raw_shots.items():
        if not isinstance(raw_shot, dict):
            raise AnnotationError(f"invalid shot {raw_id!r}")
        shot_id = str(raw_id)
        if shot_id in shots:
            raise AnnotationError(f"duplicate shot id {shot_id}")
        try:
            start, end = int(raw_shot["start_frame"]), int(raw_shot["end_frame"])
        except (KeyError, TypeError, ValueError) as error:
            raise AnnotationError(f"shot {shot_id} has invalid frame interval") from error
        if not 0 <= start < end <= frame_count:
            raise AnnotationError(f"shot {shot_id} lies outside source frame range")
        play_id = raw_shot.get("play_id")
        play_id = None if play_id is None else str(play_id)
        split = str(raw_shot.get("split", "unassigned"))
        if play_id is not None:
            previous = play_splits.setdefault(play_id, split)
            if previous != split:
                raise AnnotationError(f"play {play_id} is assigned to conflicting splits")
        shots[shot_id] = ShotAnnotation(shot_id, start, end, play_id, split)
    ordered_shots = sorted(shots.values(), key=lambda shot: (shot.start_frame, shot.end_frame, shot.shot_id))
    for previous, current in zip(ordered_shots, ordered_shots[1:]):
        if current.start_frame < previous.end_frame:
            raise AnnotationError(f"shot intervals overlap: {previous.shot_id} and {current.shot_id}")
    raw_annotations = value.get("annotations", [])
    if not isinstance(raw_annotations, list):
        raise AnnotationError("annotations must be a list")
    seen_ids: set[str] = set()
    annotations: list[dict[str, Any]] = []
    for raw in raw_annotations:
        if not isinstance(raw, dict) or not raw.get("id"):
            raise AnnotationError("each annotation needs a unique id")
        annotation_id = str(raw["id"])
        if annotation_id in seen_ids:
            raise AnnotationError(f"duplicate annotation id {annotation_id}")
        seen_ids.add(annotation_id)
        shot_id = str(raw.get("shot_id", ""))
        shot = shots.get(shot_id)
        if shot is None:
            raise AnnotationError(f"annotation {annotation_id} names an unknown shot")
        try:
            frame_index = int(raw["source_frame"])
        except (KeyError, TypeError, ValueError) as error:
            raise AnnotationError(f"annotation {annotation_id} has no source_frame") from error
        if not shot.contains(frame_index):
            raise AnnotationError(f"annotation {annotation_id} lies outside shot {shot_id}")
        if "bbox_xyxy_px" in raw:
            box = _box(raw["bbox_xyxy_px"])
            if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                raise AnnotationError(f"annotation {annotation_id} bbox lies outside source image")
        if raw.get("coordinate_space", "source") != "source":
            raise AnnotationError(f"annotation {annotation_id} must be converted to source coordinates before review")
        if require_reviewed and raw.get("review_status") not in {"reviewed", "accepted"}:
            raise AnnotationError(f"annotation {annotation_id} is not reviewed")
        annotations.append(dict(raw))
    raw_landmarks = value.get("landmarks", [])
    if not isinstance(raw_landmarks, list):
        raise AnnotationError("landmarks must be a list")
    seen_landmarks: set[str] = set()
    landmarks: list[dict[str, Any]] = []
    for raw in raw_landmarks:
        if not isinstance(raw, dict) or not raw.get("id"):
            raise AnnotationError("each landmark needs a unique id")
        landmark_id = str(raw["id"])
        if landmark_id in seen_landmarks:
            raise AnnotationError(f"duplicate landmark id {landmark_id}")
        seen_landmarks.add(landmark_id)
        shot_id = str(raw.get("shot_id", ""))
        shot = shots.get(shot_id)
        if shot is None:
            raise AnnotationError(f"landmark {landmark_id} names an unknown shot")
        try:
            frame_index = int(raw["source_frame"])
            point = tuple(float(item) for item in raw["image_xy_px"])
            field_point = tuple(float(item) for item in raw["field_xy_yards"])
        except (KeyError, TypeError, ValueError) as error:
            raise AnnotationError(f"landmark {landmark_id} is incomplete") from error
        if not shot.contains(frame_index) or len(point) != 2 or len(field_point) != 2:
            raise AnnotationError(f"landmark {landmark_id} has invalid frame or coordinate shape")
        if not all(math.isfinite(item) for item in point + field_point):
            raise AnnotationError(f"landmark {landmark_id} coordinates must be finite")
        role = str(raw.get("role", ""))
        if role not in {"fit", "withheld"}:
            raise AnnotationError(f"landmark {landmark_id} role must be fit or withheld")
        if require_reviewed and raw.get("review_status") not in {"reviewed", "accepted"}:
            raise AnnotationError(f"landmark {landmark_id} is not reviewed")
        landmarks.append(dict(raw))
    return AnnotationManifest(1, reviewed, source_sha256, width, height, frame_count, (numerator, denominator), shots, tuple(annotations), tuple(landmarks))


def source_bbox_from_crop(
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    crop_xyxy: tuple[float, float, float, float] | list[float],
    resize_xy: tuple[float, float] | list[float] | None = None,
) -> tuple[float, float, float, float]:
    """Convert a crop/resized box into original source-image coordinates."""

    box = _box(bbox_xyxy)
    crop = _box(crop_xyxy)
    if resize_xy is None:
        scale_x = scale_y = 1.0
    else:
        if len(resize_xy) != 2 or float(resize_xy[0]) <= 0 or float(resize_xy[1]) <= 0:
            raise AnnotationError("resize dimensions must be positive")
        scale_x = (crop[2] - crop[0]) / float(resize_xy[0])
        scale_y = (crop[3] - crop[1]) / float(resize_xy[1])
    return (crop[0] + box[0] * scale_x, crop[1] + box[1] * scale_y, crop[0] + box[2] * scale_x, crop[1] + box[3] * scale_y)
