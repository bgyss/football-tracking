"""Versioned, human-reviewed annotation manifest validation."""

from __future__ import annotations

import json
import math
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .field import field_landmark, validate_field_point


class AnnotationError(ValueError):
    """Raised when an annotation manifest cannot support evaluation."""


@dataclass(frozen=True, slots=True)
class ShotAnnotation:
    shot_id: str
    start_frame: int
    end_frame: int
    play_id: str | None
    split: str
    camera_label: str = "unknown"

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
    frame_labels: tuple[dict[str, Any], ...] = ()
    timing_events: tuple[dict[str, Any], ...] = ()


def _validate_review_metadata(raw: Mapping[str, Any], label: str) -> None:
    reviewer = str(raw.get("reviewer", "")).strip()
    reviewed_at = str(raw.get("reviewed_at", "")).strip()
    if not reviewer or not reviewed_at:
        raise AnnotationError(f"{label} needs reviewer and reviewed_at metadata")
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise AnnotationError(f"{label} reviewed_at must be an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise AnnotationError(f"{label} reviewed_at must include a timezone")
    try:
        revision = int(raw.get("revision", 0))
    except (TypeError, ValueError) as error:
        raise AnnotationError(f"{label} revision must be positive") from error
    if revision <= 0:
        raise AnnotationError(f"{label} revision must be positive")
    try:
        confidence = float(raw.get("annotation_confidence"))
    except (TypeError, ValueError) as error:
        raise AnnotationError(f"{label} needs annotation_confidence between 0 and 1") from error
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise AnnotationError(f"{label} annotation_confidence must be between 0 and 1")


def _validate_identity_review_metadata(raw: Mapping[str, Any], label: str) -> None:
    status = str(raw.get("cross_shot_review_status", "")).strip()
    raw_global_id = raw.get("global_id", raw.get("cross_shot_id"))
    global_id = str(raw_global_id).strip() if raw_global_id is not None else ""
    meaningful_id = global_id.lower() not in {"", "unknown", "ambiguous", "unresolved"}
    if meaningful_id and status != "approved":
        raise AnnotationError(f"{label} global_id requires cross_shot_review_status: approved")
    if not status:
        return
    if status not in {"approved", "ambiguous", "rejected"}:
        raise AnnotationError(f"{label} has unsupported cross_shot_review_status")
    if status == "approved" and not meaningful_id:
        raise AnnotationError(f"{label} approved cross-shot review needs a global_id")
    if status in {"ambiguous", "rejected"} and meaningful_id:
        raise AnnotationError(f"{label} {status} cross-shot review cannot assign a global_id")
    second_reviewer = str(raw.get("identity_second_reviewer", "")).strip()
    if not second_reviewer or second_reviewer.casefold() == str(raw.get("reviewer", "")).strip().casefold():
        raise AnnotationError(f"{label} cross-shot decision needs an independent second reviewer")
    second_reviewed_at = str(raw.get("identity_second_reviewed_at", "")).strip()
    try:
        timestamp = datetime.fromisoformat(second_reviewed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise AnnotationError(f"{label} identity_second_reviewed_at must be an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise AnnotationError(f"{label} identity_second_reviewed_at must include a timezone")
    try:
        second_revision = int(raw.get("identity_second_revision", 0))
        second_confidence = float(raw.get("identity_second_confidence"))
    except (TypeError, ValueError) as error:
        raise AnnotationError(f"{label} second-review metadata is incomplete") from error
    if second_revision <= 0:
        raise AnnotationError(f"{label} identity_second_revision must be positive")
    if not math.isfinite(second_confidence) or not 0.0 <= second_confidence <= 1.0:
        raise AnnotationError(f"{label} identity_second_confidence must be between zero and one")


def _box(value: Any) -> tuple[float, float, float, float]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise AnnotationError("bbox_xyxy_px must contain four numeric values") from error
    if len(result) != 4 or not all(math.isfinite(item) for item in result) or result[2] <= result[0] or result[3] <= result[1]:
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
        time_base_value = source["time_base"]
        if len(time_base_value) != 2:  # type: ignore[arg-type]
            raise AnnotationError("source time_base must contain exactly two values")
        numerator, denominator = int(time_base_value[0]), int(time_base_value[1])  # type: ignore[index]
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
        camera_label = str(raw_shot.get("camera_label", ""))
        if require_reviewed and not camera_label.strip():
            raise AnnotationError(f"shot {shot_id} needs camera_label metadata")
        if play_id is not None:
            previous = play_splits.setdefault(play_id, split)
            if previous != split:
                raise AnnotationError(f"play {play_id} is assigned to conflicting splits")
        shots[shot_id] = ShotAnnotation(shot_id, start, end, play_id, split, camera_label or "unknown")
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
            pts = int(raw["pts"])
        except (KeyError, TypeError, ValueError) as error:
            raise AnnotationError(f"annotation {annotation_id} has no source_frame or pts") from error
        if not shot.contains(frame_index):
            raise AnnotationError(f"annotation {annotation_id} lies outside shot {shot_id}")
        if pts < 0:
            raise AnnotationError(f"annotation {annotation_id} has invalid pts")
        if "bbox_xyxy_px" in raw:
            box = _box(raw["bbox_xyxy_px"])
            if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                raise AnnotationError(f"annotation {annotation_id} bbox lies outside source image")
        if "ground_contact_xy_px" in raw:
            try:
                contact_px = tuple(float(item) for item in raw["ground_contact_xy_px"])
            except (TypeError, ValueError) as error:
                raise AnnotationError(f"annotation {annotation_id} ground_contact_xy_px must contain numeric coordinates") from error
            if len(contact_px) != 2 or not all(math.isfinite(item) for item in contact_px) or not 0.0 <= contact_px[0] <= width or not 0.0 <= contact_px[1] <= height:
                raise AnnotationError(f"annotation {annotation_id} ground_contact_xy_px lies outside source image")
            if require_reviewed and raw.get("ground_contact_confidence") is None:
                raise AnnotationError(f"annotation {annotation_id} needs ground_contact_confidence for a reviewed contact point")
        if raw.get("ground_contact_confidence") is not None:
            try:
                contact_confidence = float(raw["ground_contact_confidence"])
            except (TypeError, ValueError) as error:
                raise AnnotationError(f"annotation {annotation_id} has invalid ground_contact_confidence") from error
            if not math.isfinite(contact_confidence) or not 0.0 <= contact_confidence <= 1.0:
                raise AnnotationError(f"annotation {annotation_id} ground_contact_confidence must be between zero and one")
        if raw.get("coordinate_space", "source") != "source":
            raise AnnotationError(f"annotation {annotation_id} must be converted to source coordinates before review")
        if require_reviewed and raw.get("review_status") not in {"reviewed", "accepted"}:
            raise AnnotationError(f"annotation {annotation_id} is not reviewed")
        label = str(raw.get("label", "player"))
        if label not in {"player", "official", "football"}:
            raise AnnotationError(f"annotation {annotation_id} has unsupported label {label!r}")
        visibility = raw.get("visibility")
        if visibility is not None and visibility not in {"visible", "partially_visible", "occluded", "out_of_frame", "unknown"}:
            raise AnnotationError(f"annotation {annotation_id} has unsupported visibility {visibility!r}")
        team = raw.get("team")
        if team is not None and not str(team).strip():
            raise AnnotationError(f"annotation {annotation_id} has an empty team")
        if require_reviewed:
            _validate_review_metadata(raw, f"annotation {annotation_id}")
            _validate_identity_review_metadata(raw, f"annotation {annotation_id}")
        annotations.append(dict(raw))
    identity_decisions: dict[tuple[str, str], tuple[str, str]] = {}
    for raw in annotations:
        status = str(raw.get("cross_shot_review_status", "")).strip()
        raw_global_id = raw.get("global_id", raw.get("cross_shot_id"))
        if not status and raw_global_id is None:
            continue
        object_id = str(raw.get("track_id") or raw.get("id") or "").strip()
        decision = (status, str(raw_global_id or "").strip().lower())
        key = (str(raw["shot_id"]), object_id)
        previous = identity_decisions.setdefault(key, decision)
        if previous != decision:
            raise AnnotationError(f"track {key[0]}:{key[1]} has conflicting cross-shot review decisions")
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
            pts = int(raw["pts"])
            point = tuple(float(item) for item in raw["image_xy_px"])
            field_point = tuple(float(item) for item in raw["field_xy_yards"])
        except (KeyError, TypeError, ValueError) as error:
            raise AnnotationError(f"landmark {landmark_id} is incomplete") from error
        if not shot.contains(frame_index) or len(point) != 2 or len(field_point) != 2:
            raise AnnotationError(f"landmark {landmark_id} has invalid frame or coordinate shape")
        if pts < 0:
            raise AnnotationError(f"landmark {landmark_id} has invalid pts")
        if not all(math.isfinite(item) for item in point + field_point):
            raise AnnotationError(f"landmark {landmark_id} coordinates must be finite")
        if not 0.0 <= point[0] <= width or not 0.0 <= point[1] <= height:
            raise AnnotationError(f"landmark {landmark_id} lies outside source image")
        try:
            validate_field_point(field_point)
        except ValueError as error:
            raise AnnotationError(f"landmark {landmark_id} lies outside the canonical field") from error
        semantic_id = raw.get("landmark_id")
        if semantic_id is not None:
            try:
                expected = field_landmark(str(semantic_id))
            except ValueError as error:
                raise AnnotationError(f"landmark {landmark_id} has invalid semantic landmark_id") from error
            if math.dist(field_point, expected) > 0.01:
                raise AnnotationError(f"landmark {landmark_id} field coordinate disagrees with landmark_id")
        role = str(raw.get("role", ""))
        if role not in {"fit", "withheld"}:
            raise AnnotationError(f"landmark {landmark_id} role must be fit or withheld")
        if require_reviewed and raw.get("review_status") not in {"reviewed", "accepted"}:
            raise AnnotationError(f"landmark {landmark_id} is not reviewed")
        if require_reviewed:
            _validate_review_metadata(raw, f"landmark {landmark_id}")
        landmarks.append(dict(raw))
    raw_frame_labels = value.get("frame_labels", [])
    if not isinstance(raw_frame_labels, list):
        raise AnnotationError("frame_labels must be a list")
    seen_frame_labels: set[tuple[str, int]] = set()
    frame_labels: list[dict[str, Any]] = []
    for raw in raw_frame_labels:
        if not isinstance(raw, dict):
            raise AnnotationError("each frame label must be an object")
        shot_id = str(raw.get("shot_id", ""))
        shot = shots.get(shot_id)
        if shot is None:
            raise AnnotationError("frame label names an unknown shot")
        try:
            frame_index = int(raw["source_frame"])
            pts = int(raw["pts"])
        except (KeyError, TypeError, ValueError) as error:
            raise AnnotationError("frame label needs source_frame and pts") from error
        if not shot.contains(frame_index) or pts < 0:
            raise AnnotationError("frame label lies outside its shot or has invalid pts")
        key = (shot_id, frame_index)
        if key in seen_frame_labels:
            raise AnnotationError(f"duplicate frame label {shot_id}:{frame_index}")
        seen_frame_labels.add(key)
        if require_reviewed and raw.get("review_status") not in {"reviewed", "accepted"}:
            raise AnnotationError(f"frame label {shot_id}:{frame_index} is not reviewed")
        if require_reviewed:
            _validate_review_metadata(raw, f"frame label {shot_id}:{frame_index}")
        frame_labels.append(dict(raw))
    raw_timing_events = value.get("timing_events", [])
    if not isinstance(raw_timing_events, list):
        raise AnnotationError("timing_events must be a list")
    seen_timing_events: set[str] = set()
    timing_events: list[dict[str, Any]] = []
    for raw in raw_timing_events:
        if not isinstance(raw, dict) or not raw.get("id"):
            raise AnnotationError("each timing event needs a unique id")
        event_id = str(raw["id"])
        if event_id in seen_timing_events:
            raise AnnotationError(f"duplicate timing event id {event_id}")
        seen_timing_events.add(event_id)
        shot_id = str(raw.get("shot_id", ""))
        shot = shots.get(shot_id)
        if shot is None:
            raise AnnotationError(f"timing event {event_id} names an unknown shot")
        try:
            frame_index = int(raw["source_frame"])
            pts = int(raw["pts"])
        except (KeyError, TypeError, ValueError) as error:
            raise AnnotationError(f"timing event {event_id} needs source_frame and pts") from error
        if not shot.contains(frame_index) or pts < 0:
            raise AnnotationError(f"timing event {event_id} lies outside its shot or has invalid pts")
        if not str(raw.get("event", "")).strip():
            raise AnnotationError(f"timing event {event_id} needs an event name")
        play_time = raw.get("play_time_s")
        if play_time is not None:
            try:
                play_time = float(play_time)
            except (TypeError, ValueError) as error:
                raise AnnotationError(f"timing event {event_id} has invalid play_time_s") from error
            if not math.isfinite(play_time):
                raise AnnotationError(f"timing event {event_id} has invalid play_time_s")
        if require_reviewed:
            if raw.get("review_status") not in {"reviewed", "accepted"}:
                raise AnnotationError(f"timing event {event_id} is not reviewed")
            _validate_review_metadata(raw, f"timing event {event_id}")
        timing_events.append(dict(raw))
    return AnnotationManifest(
        1, reviewed, source_sha256, width, height, frame_count,
        (numerator, denominator), shots, tuple(annotations), tuple(landmarks),
        tuple(frame_labels), tuple(timing_events),
    )


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
        if len(resize_xy) != 2 or not all(math.isfinite(float(item)) for item in resize_xy) or float(resize_xy[0]) <= 0 or float(resize_xy[1]) <= 0:
            raise AnnotationError("resize dimensions must be positive")
        scale_x = (crop[2] - crop[0]) / float(resize_xy[0])
        scale_y = (crop[3] - crop[1]) / float(resize_xy[1])
    return (crop[0] + box[0] * scale_x, crop[1] + box[1] * scale_y, crop[0] + box[2] * scale_x, crop[1] + box[3] * scale_y)


def manifest_template_from_review_pack(pack: Mapping[str, Any], shots: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Create an unreviewed annotation-manifest skeleton from a review pack."""

    if not isinstance(pack, Mapping) or int(pack.get("schema_version", 0)) != 1:
        raise AnnotationError("review pack schema_version must be 1")
    source = pack.get("source")
    if not isinstance(source, Mapping) or not source.get("sha256"):
        raise AnnotationError("review pack has no source hash")
    normalized_shots: dict[str, dict[str, Any]] = {}
    for shot_id, raw in sorted(shots.items()):
        if not isinstance(raw, Mapping):
            raise AnnotationError(f"invalid shot template {shot_id}")
        try:
            start, end = int(raw["start_frame"]), int(raw["end_frame"])
        except (KeyError, TypeError, ValueError) as error:
            raise AnnotationError(f"shot template {shot_id} needs start_frame/end_frame") from error
        if start < 0 or end <= start or end > int(source["frame_count"]):
            raise AnnotationError(f"shot template {shot_id} has an invalid frame range")
        camera_label = str(raw.get("camera_label", ""))
        if not camera_label:
            raise AnnotationError(f"shot template {shot_id} needs camera_label")
        normalized_shots[str(shot_id)] = {"start_frame": start, "end_frame": end, "play_id": None if raw.get("play_id") is None else str(raw["play_id"]), "split": str(raw.get("split", "unassigned")), "camera_label": camera_label}
    ordered = sorted(normalized_shots.items(), key=lambda item: (item[1]["start_frame"], item[1]["end_frame"], item[0]))
    for previous, current in zip(ordered, ordered[1:]):
        if current[1]["start_frame"] < previous[1]["end_frame"]:
            raise AnnotationError(f"shot template ranges overlap: {previous[0]} and {current[0]}")
    return {"schema_version": 1, "reviewed": False, "source": dict(source), "shots": normalized_shots, "annotations": [], "landmarks": [], "timing_events": [], "frame_labels": [], "review_frames": list(pack.get("frames", [])), "review_policy": "Fill and review all labels, then set reviewed=true."}


def mot_reference_from_manifest(manifest: AnnotationManifest, calibration_timeline: Any = None) -> dict[str, Any]:
    """Convert a reviewed annotation manifest into the evaluator reference contract.

    Player annotations use ``track_id`` (falling back to ``id``) as the
    sequence-local object ID. Optional ``global_id``/``cross_shot_id`` fields
    become the reviewed cross-shot map. This conversion never invents empty
    frames or identities: only explicitly reviewed annotation records enter the
    reference artifact.
    """

    if not manifest.reviewed:
        raise AnnotationError("cannot build an evaluator reference from an unreviewed manifest")
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {shot_id: {} for shot_id in manifest.shots}
    frame_records: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in manifest.frame_labels:
        key = (str(raw["shot_id"]), int(raw["source_frame"]))
        frame_records[key] = {"labeled": bool(raw.get("labeled", True)), "ignore": bool(raw.get("ignore", False)), "pts": int(raw["pts"])}
    cross_shot_identity: dict[str, dict[str, str]] = {}
    frame_pts: dict[tuple[str, int], int] = {}
    seen_objects: set[tuple[str, int, str]] = set()
    for raw in manifest.annotations:
        if str(raw.get("label", "player")) != "player":
            continue
        shot_id = str(raw["shot_id"])
        frame_index = int(raw["source_frame"])
        pts = int(raw["pts"])
        frame_key = (shot_id, frame_index)
        if frame_key in frame_pts and frame_pts[frame_key] != pts:
            raise AnnotationError(f"reviewed frame {shot_id}:{frame_index} has conflicting PTS values")
        frame_pts[frame_key] = pts
        object_id = str(raw.get("track_id") or raw.get("id") or "").strip()
        if not object_id:
            raise AnnotationError(f"annotation {raw.get('id', '<unknown>')} needs track_id or id")
        key = (shot_id, frame_index, object_id)
        if key in seen_objects:
            raise AnnotationError(f"duplicate reviewed object {shot_id}:{frame_index}:{object_id}")
        seen_objects.add(key)
        if "bbox_xyxy_px" not in raw:
            continue
        object_value: dict[str, Any] = {"id": object_id, "bbox_xyxy": list(_box(raw["bbox_xyxy_px"]))}
        if raw.get("team") is not None:
            object_value["team"] = str(raw["team"])
        if raw.get("visibility") is not None:
            object_value["visibility"] = str(raw["visibility"])
        if raw.get("occluded") is not None:
            object_value["occluded"] = bool(raw["occluded"])
        contact = raw.get("ground_contact_xy_yards")
        if contact is not None:
            try:
                object_value["ground_contact_xy_yards"] = list(validate_field_point(tuple(float(item) for item in contact)))
            except (TypeError, ValueError) as error:
                raise AnnotationError(f"annotation {raw.get('id', '<unknown>')} has invalid ground contact") from error
            if raw.get("ground_contact_confidence") is not None:
                try:
                    object_value["ground_contact_confidence"] = float(raw["ground_contact_confidence"])
                except (TypeError, ValueError) as error:
                    raise AnnotationError(f"annotation {raw.get('id', '<unknown>')} has invalid ground contact confidence") from error
            object_value["ground_contact_projection_status"] = "reviewed_field_coordinate"
        elif raw.get("ground_contact_xy_px") is not None:
            object_value["ground_contact_projection_status"] = "not_requested" if calibration_timeline is None else "calibration_unavailable"
            if raw.get("ground_contact_confidence") is not None:
                object_value["ground_contact_confidence"] = float(raw["ground_contact_confidence"])
            if calibration_timeline is not None:
                estimate = calibration_timeline.at(shot_id, pts)
                if estimate is not None:
                    object_value["ground_contact_projection_status"] = estimate.status
                    if estimate.identity_eligible:
                        try:
                            import cv2
                            import numpy as np

                            point_xy = tuple(float(item) for item in raw["ground_contact_xy_px"])
                            if len(estimate.support_polygon_px) < 3:
                                object_value["ground_contact_projection_status"] = "calibration_support_missing"
                                point_xy = ()
                            else:
                                polygon = np.asarray(estimate.support_polygon_px, dtype=np.float32).reshape(-1, 1, 2)
                                if cv2.pointPolygonTest(polygon, point_xy, False) < 0:
                                    object_value["ground_contact_projection_status"] = "outside_calibration_support"
                                    point_xy = ()
                            if point_xy:
                                from .calibration import ImagePoint

                                projected = estimate.homography.project(ImagePoint(*point_xy))
                                if projected is None:
                                    object_value["ground_contact_projection_status"] = "projection_invalid"
                                else:
                                    object_value["ground_contact_xy_yards"] = [projected.x_yards, projected.y_yards]
                                    object_value["ground_contact_projection_status"] = "projected"
                                    object_value["calibration_id"] = estimate.calibration_id
                        except (TypeError, ValueError) as error:
                            raise AnnotationError(f"annotation {raw.get('id', '<unknown>')} has invalid ground contact pixel coordinates") from error
        grouped.setdefault(shot_id, {}).setdefault(frame_index, []).append(object_value)
        global_id = raw.get("global_id", raw.get("cross_shot_id"))
        if global_id is not None and str(global_id).strip().lower() not in {"", "unknown", "ambiguous", "unresolved"}:
            normalized_global_id = str(global_id)
            previous_global_id = cross_shot_identity.setdefault(shot_id, {}).get(object_id)
            if previous_global_id is not None and previous_global_id != normalized_global_id:
                raise AnnotationError(f"reviewed object {shot_id}:{object_id} has conflicting global identities")
            cross_shot_identity[shot_id][object_id] = normalized_global_id
    sequences: dict[str, dict[str, Any]] = {}
    for shot_id, frames in sorted(grouped.items()):
        frame_indices = sorted(set(frames) | {frame_index for frame_shot, frame_index in frame_records if frame_shot == shot_id})
        sequences[shot_id] = {"frames": {str(frame_index): {**frame_records.get((shot_id, frame_index), {"labeled": True, "ignore": False, "pts": frame_pts.get((shot_id, frame_index))}), "objects": objects} for frame_index, objects in ((frame_index, frames.get(frame_index, [])) for frame_index in frame_indices)}}
    reference: dict[str, Any] = {"schema_version": 1, "reviewed": True, "source_sha256": manifest.source_sha256, "sequences": sequences}
    if cross_shot_identity:
        reference["cross_shot_identity"] = cross_shot_identity
    return reference
