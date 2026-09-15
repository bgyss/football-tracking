"""Frame/PTS-scoped calibration with explicit support intervals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import cv2

from .calibration import FieldPoint, Homography, ImagePoint
from .field import field_landmark, validate_field_point


class CalibrationTimelineError(ValueError):
    """Raised when a time-keyed calibration cannot be used safely."""


@dataclass(frozen=True, slots=True)
class CalibrationEstimate:
    shot_id: str
    calibration_id: str
    homography: Homography
    pts_start: int
    pts_end: int | None
    source_keyframe_ids: tuple[str, ...]
    status: str
    reason: str | None = None
    support_polygon_px: tuple[tuple[float, float], ...] = ()

    def contains(self, pts: int) -> bool:
        return pts >= self.pts_start and (self.pts_end is None or pts < self.pts_end)

    @property
    def identity_eligible(self) -> bool:
        return self.status == "valid" and self.homography.identity_eligible


class CalibrationTimeline:
    def __init__(self, estimates: Mapping[str, Sequence[CalibrationEstimate]]) -> None:
        normalized: dict[str, tuple[CalibrationEstimate, ...]] = {}
        for shot_id, values in estimates.items():
            raw_ordered = tuple(sorted(values, key=lambda estimate: (estimate.pts_start, estimate.calibration_id)))
            # A keyframe without an explicit end owns the interval up to the
            # next keyframe. The final keyframe may remain open-ended.
            ordered = tuple(
                replace(estimate, pts_end=raw_ordered[index + 1].pts_start)
                if estimate.pts_end is None and index + 1 < len(raw_ordered)
                else estimate
                for index, estimate in enumerate(raw_ordered)
            )
            previous_end: int | None = None
            for estimate in ordered:
                if estimate.shot_id != str(shot_id):
                    raise CalibrationTimelineError("estimate shot_id does not match its mapping key")
                if estimate.status not in {"valid", "unvalidated", "invalid", "partial"}:
                    raise CalibrationTimelineError(f"unsupported calibration status {estimate.status!r}")
                if estimate.status == "valid" and not estimate.homography.identity_eligible:
                    raise CalibrationTimelineError(f"calibration {estimate.calibration_id} is marked valid without withheld eligibility")
                if estimate.pts_start < 0 or (estimate.pts_end is not None and estimate.pts_end <= estimate.pts_start):
                    raise CalibrationTimelineError("calibration support interval is invalid")
                if previous_end is not None and estimate.pts_start < previous_end:
                    raise CalibrationTimelineError(f"overlapping calibration intervals for {shot_id}")
                previous_end = estimate.pts_end
            normalized[str(shot_id)] = ordered
        self._estimates = normalized

    def at(self, shot_id: str, pts: int) -> CalibrationEstimate | None:
        if pts < 0:
            raise ValueError("pts must be non-negative")
        for estimate in self._estimates.get(str(shot_id), ()):
            if estimate.contains(pts):
                return estimate
        return None

    def shots(self) -> tuple[str, ...]:
        return tuple(sorted(self._estimates))

    def identity_eligible_shots(self) -> tuple[str, ...]:
        return tuple(sorted(shot_id for shot_id, values in self._estimates.items() if any(value.identity_eligible for value in values)))

    def to_dict(self) -> dict[str, object]:
        shots: dict[str, object] = {}
        for shot_id, values in sorted(self._estimates.items()):
            shots[shot_id] = {
                "keyframes": [
                    {
                        "calibration_id": estimate.calibration_id,
                        "pts_start": estimate.pts_start,
                        "pts_end": estimate.pts_end,
                        "source_keyframe_ids": list(estimate.source_keyframe_ids),
                        "status": estimate.status,
                        "reason": estimate.reason,
                        "support_polygon_px": estimate.support_polygon_px,
                        "homography": {
                            "matrix": estimate.homography.matrix,
                            "fit_error_px": estimate.homography.fit_error_px,
                            "max_error_px": estimate.homography.max_error_px,
                            "median_error_yards": estimate.homography.median_error_yards,
                            "max_error_yards": estimate.homography.max_error_yards,
                            "withheld_median_error_yards": estimate.homography.withheld_median_error_yards,
                            "withheld_p95_error_yards": estimate.homography.withheld_p95_error_yards,
                            "withheld_point_count": estimate.homography.withheld_point_count,
                            "withheld_errors_yards": estimate.homography.withheld_errors_yards,
                            "inlier_count": estimate.homography.inlier_count,
                            "point_count": estimate.homography.point_count,
                            "reprojection_threshold_px": estimate.homography.reprojection_threshold_px,
                            "calibration_id": estimate.homography.calibration_id,
                            "status": estimate.homography.status,
                            "reason": estimate.homography.reason,
                        },
                    }
                    for estimate in values
                ]
            }
        statuses = {estimate.status for values in self._estimates.values() for estimate in values}
        overall_status = "valid" if statuses == {"valid"} else "invalid" if statuses == {"invalid"} else "partial" if "invalid" in statuses or "partial" in statuses or len(statuses) > 1 else "unvalidated"
        return {"schema_version": 2, "status": overall_status, "shots": shots}


def _points(config: Mapping[str, object]) -> tuple[list[ImagePoint], list[FieldPoint], list[ImagePoint] | None, list[FieldPoint] | None]:
    try:
        image = [ImagePoint(float(point[0]), float(point[1])) for point in config["image_points"]]  # type: ignore[index]
        field = [FieldPoint(float(point[0]), float(point[1])) for point in config["field_points"]]  # type: ignore[index]
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise CalibrationTimelineError(f"invalid calibration landmarks: {error}") from error
    withheld_image_raw = config.get("withheld_image_points")
    withheld_field_raw = config.get("withheld_field_points")
    if (withheld_image_raw is None) != (withheld_field_raw is None):
        raise CalibrationTimelineError("withheld image and field points must be paired")
    try:
        withheld_image = None if withheld_image_raw is None else [ImagePoint(float(point[0]), float(point[1])) for point in withheld_image_raw]  # type: ignore[index]
        withheld_field = None if withheld_field_raw is None else [FieldPoint(float(point[0]), float(point[1])) for point in withheld_field_raw]  # type: ignore[index]
    except (TypeError, ValueError, IndexError) as error:
        raise CalibrationTimelineError(f"invalid withheld calibration landmarks: {error}") from error
    return image, field, withheld_image, withheld_field


def load_calibration_timeline(path: str | Path, *, source_sha256: str | None = None) -> CalibrationTimeline:
    """Load schema v2 time-keyed fits, or wrap legacy shot fits as unscoped estimates."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationTimelineError(f"unable to read calibration timeline: {error}") from error
    if not isinstance(value, dict):
        raise CalibrationTimelineError("calibration must be an object")
    try:
        schema_version = int(value.get("schema_version", 1))
    except (TypeError, ValueError) as error:
        raise CalibrationTimelineError("calibration schema_version must be numeric") from error
    if schema_version not in {1, 2}:
        raise CalibrationTimelineError("unsupported calibration schema_version")
    declared_source = value.get("source_sha256")
    if source_sha256 is not None and schema_version >= 2 and not declared_source:
        raise CalibrationTimelineError("schema-v2 calibration requires source_sha256")
    if source_sha256 is not None and declared_source is not None and str(declared_source) != source_sha256:
        raise CalibrationTimelineError("calibration source sha256 does not match the input video")
    raw_shots = value.get("shots")
    if not isinstance(raw_shots, dict) or not raw_shots:
        # Legacy shared fit is deliberately represented under '*' and has no
        # shot-specific identity eligibility.
        if "image_points" not in value:
            raise CalibrationTimelineError("calibration has no shots")
        raw_shots = {"*": {"keyframes": [value]}}
    result: dict[str, list[CalibrationEstimate]] = {}
    for shot_id, raw in raw_shots.items():
        if not isinstance(raw, dict):
            raise CalibrationTimelineError(f"invalid calibration shot {shot_id!r}")
        keyframes = raw.get("keyframes")
        if keyframes is None and "image_points" in raw:
            keyframes = [raw]
        if not isinstance(keyframes, list) or not keyframes:
            raise CalibrationTimelineError(f"shot {shot_id} has no keyframes")
        estimates: list[CalibrationEstimate] = []
        for index, config in enumerate(keyframes):
            if not isinstance(config, dict):
                raise CalibrationTimelineError(f"invalid keyframe {shot_id}:{index}")
            serialized = config.get("homography")
            if isinstance(serialized, dict) and "matrix" in serialized:
                try:
                    matrix = tuple(tuple(float(item) for item in row) for row in serialized["matrix"])
                    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                        raise ValueError("matrix must be 3x3")
                    homography = Homography(
                        matrix, float(serialized.get("fit_error_px", serialized.get("median_error_px", float("inf")))), float(serialized.get("max_error_px", float("inf"))), int(serialized["inlier_count"]), int(serialized["point_count"]), float(serialized.get("reprojection_threshold_px", 3.0)), str(serialized.get("calibration_id", f"serialized-{shot_id}-{index}")), float(serialized.get("median_error_yards", float("inf"))), float(serialized.get("max_error_yards", float("inf"))), None if serialized.get("withheld_median_error_yards") is None else float(serialized["withheld_median_error_yards"]), None if serialized.get("withheld_p95_error_yards") is None else float(serialized["withheld_p95_error_yards"]), int(serialized.get("withheld_point_count", 0)), str(serialized.get("status", config.get("status", "unvalidated"))), serialized.get("reason") if isinstance(serialized.get("reason"), str) else None, tuple(float(item) for item in serialized.get("withheld_errors_yards", ())),
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise CalibrationTimelineError(f"invalid serialized homography {shot_id}:{index}") from error
            else:
                image, field, withheld_image, withheld_field = _points(config)
                homography = Homography.fit(
                    image,
                    field,
                    float(config.get("reprojection_threshold_px", 3.0)),
                    withheld_image_points=withheld_image,
                    withheld_field_points=withheld_field,
                )
            try:
                pts_start = int(config.get("pts_start", 0))
                pts_end_value = config.get("pts_end")
                pts_end = None if pts_end_value is None else int(pts_end_value)
            except (TypeError, ValueError) as error:
                raise CalibrationTimelineError(f"invalid calibration interval {shot_id}:{index}") from error
            keyframe_id = str(config.get("keyframe_id", f"{shot_id}:k{index}"))
            raw_polygon = config.get("support_polygon_px") or ()
            try:
                support_polygon = tuple((float(point[0]), float(point[1])) for point in raw_polygon)  # type: ignore[index]
            except (TypeError, ValueError, IndexError) as error:
                raise CalibrationTimelineError(f"invalid support polygon {shot_id}:{index}") from error
            if support_polygon and (len(support_polygon) < 3 or not np.all(np.isfinite(np.asarray(support_polygon, dtype=float)))):
                raise CalibrationTimelineError("support polygon needs at least three finite points")
            # A timeline fit without withheld points remains mathematically
            # projectable but is not a promoted calibration result.
            status = str(config.get("status", homography.status))
            if status not in {"valid", "unvalidated", "invalid", "partial"}:
                raise CalibrationTimelineError(f"unsupported calibration status {status!r}")
            if status == "valid" and not homography.identity_eligible:
                raise CalibrationTimelineError(f"calibration {shot_id}:{index} is marked valid without withheld eligibility")
            calibration_id = hashlib.sha256(f"{shot_id}:{keyframe_id}:{homography.calibration_id}".encode()).hexdigest()[:16]
            estimates.append(CalibrationEstimate(str(shot_id), calibration_id, homography, pts_start, pts_end, (keyframe_id,), status, config.get("reason") if isinstance(config.get("reason"), str) else homography.reason, support_polygon))
        result[str(shot_id)] = estimates
    return CalibrationTimeline(result)


def timeline_from_landmark_records(
    landmarks: Sequence[Mapping[str, object]],
) -> CalibrationTimeline:
    """Fit a reviewed timeline directly from manifest landmark records.

    Records must carry ``shot_id``, ``source_pts``,
    ``image_xy_px``, ``field_xy_yards`` and a ``role`` of ``fit`` or
    ``withheld``. A keyframe is valid only when it has at least four fit points
    and at least one withheld point; intervals are bounded by the next reviewed
    keyframe in the same shot.
    """

    grouped: dict[tuple[str, int], dict[str, list[Mapping[str, object]]]] = {}
    seen_ids: set[str] = set()
    for raw in landmarks:
        if not isinstance(raw, Mapping):
            raise CalibrationTimelineError("landmark record must be an object")
        try:
            landmark_id = str(raw["id"])
            shot_id = str(raw["shot_id"])
            pts = int(raw["source_pts"])
            role = str(raw["role"])
        except (KeyError, TypeError, ValueError) as error:
            raise CalibrationTimelineError("landmark record is missing shot/time/role") from error
        if landmark_id in seen_ids:
            raise CalibrationTimelineError(f"duplicate landmark id {landmark_id}")
        seen_ids.add(landmark_id)
        if role not in {"fit", "withheld"} or pts < 0:
            raise CalibrationTimelineError("landmark role or time is invalid")
        semantic_id = raw.get("landmark_id")
        if semantic_id is not None:
            try:
                expected_field = field_landmark(str(semantic_id))
                supplied_field = tuple(float(item) for item in raw["field_xy_yards"])  # type: ignore[index]
            except (KeyError, TypeError, ValueError) as error:
                raise CalibrationTimelineError(f"invalid semantic landmark {semantic_id!r}") from error
            if len(supplied_field) != 2 or float(np.linalg.norm(np.asarray(supplied_field) - np.asarray(expected_field))) > 0.01:
                raise CalibrationTimelineError(f"field coordinates do not match semantic landmark {semantic_id!r}")
        grouped.setdefault((shot_id, pts), {"fit": [], "withheld": []})[role].append(raw)
    estimates: dict[str, list[CalibrationEstimate]] = {}
    for (shot_id, pts), roles in sorted(grouped.items()):
        if len(roles["fit"]) < 4 or not roles["withheld"]:
            raise CalibrationTimelineError(f"keyframe {shot_id}@{pts} needs four fit and one withheld landmark")
        try:
            image = [ImagePoint(float(record["image_xy_px"][0]), float(record["image_xy_px"][1])) for record in roles["fit"]]  # type: ignore[index]
            field = [FieldPoint(float(record["field_xy_yards"][0]), float(record["field_xy_yards"][1])) for record in roles["fit"]]  # type: ignore[index]
            withheld_image = [ImagePoint(float(record["image_xy_px"][0]), float(record["image_xy_px"][1])) for record in roles["withheld"]]  # type: ignore[index]
            withheld_field = [FieldPoint(float(record["field_xy_yards"][0]), float(record["field_xy_yards"][1])) for record in roles["withheld"]]  # type: ignore[index]
            for point in field + withheld_field:
                validate_field_point((point.x_yards, point.y_yards))
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise CalibrationTimelineError(f"invalid landmark coordinates for {shot_id}@{pts}") from error
        homography = Homography.fit(image, field, withheld_image_points=withheld_image, withheld_field_points=withheld_field)
        keyframe_id = f"{shot_id}:pts-{pts}"
        hull = cv2.convexHull(np.asarray([(point.x_px, point.y_px) for point in image], dtype=np.float32)).reshape(-1, 2)
        support_polygon = tuple((float(point[0]), float(point[1])) for point in hull) if len(hull) >= 3 else ()
        estimate = CalibrationEstimate(shot_id, keyframe_id, homography, pts, None, (keyframe_id,), "valid" if homography.identity_eligible else "invalid", homography.reason, support_polygon)
        estimates.setdefault(shot_id, []).append(estimate)
    if not estimates:
        raise CalibrationTimelineError("no landmark records supplied")
    return CalibrationTimeline(estimates)


def timeline_quality_report(timeline: CalibrationTimeline) -> dict[str, object]:
    """Report fit/withheld errors for every supported shot interval."""

    values: dict[str, dict[str, object]] = {}
    errors: list[float] = []
    for shot_id in timeline.shots():
        entries = []
        for estimate in timeline._estimates.get(shot_id, ()):
            estimate_errors = list(estimate.homography.withheld_errors_yards)
            errors.extend(estimate_errors)
            entries.append({"calibration_id": estimate.calibration_id, "pts_start": estimate.pts_start, "pts_end": estimate.pts_end, "status": estimate.status, "fit_error_px": estimate.homography.fit_error_px, "max_error_px": estimate.homography.max_error_px, "median_fit_error_yards": estimate.homography.median_error_yards, "withheld_point_count": estimate.homography.withheld_point_count, "median_error_yards": estimate.homography.withheld_median_error_yards, "p95_error_yards": estimate.homography.withheld_p95_error_yards, "reason": estimate.reason})
        values[shot_id] = {"keyframes": entries}
    if not errors:
        statuses = {entry["status"] for shot in values.values() for entry in shot["keyframes"]}
        return {"status": "invalid" if "invalid" in statuses else "unvalidated", "shots": values, "reason": "no withheld landmark errors"}
    median = float(np.median(np.asarray(errors, dtype=float)))
    p95 = float(np.percentile(np.asarray(errors, dtype=float), 95))
    return {"status": "valid" if median <= 1.0 and p95 <= 2.0 and all(entry["status"] == "valid" for shot in values.values() for entry in shot["keyframes"]) else "invalid", "shots": values, "withheld_point_count": len(errors), "median_error_yards": median, "p95_error_yards": p95, "gate": {"median_at_most_1_yard": median <= 1.0, "p95_at_most_2_yards": p95 <= 2.0}}


def propagate_homography(keyframe: Homography, image_motion: Sequence[Sequence[float]]) -> Homography:
    """Compose a keyframe image->field transform with G(current<-keyframe)."""

    motion = np.asarray(image_motion, dtype=np.float64)
    if motion.shape != (3, 3) or not np.all(np.isfinite(motion)):
        raise ValueError("image motion must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(motion))) <= 1e-12:
        raise ValueError("image motion must be invertible")
    matrix = np.asarray(keyframe.matrix, dtype=np.float64) @ np.linalg.inv(motion)
    if abs(float(matrix[2, 2])) <= 1e-12:
        raise ValueError("propagated homography has invalid scale")
    matrix /= matrix[2, 2]
    digest = hashlib.sha256(matrix.tobytes()).hexdigest()[:16]
    return Homography(
        tuple(tuple(float(item) for item in row) for row in matrix),
        keyframe.median_error_px,
        keyframe.max_error_px,
        keyframe.inlier_count,
        keyframe.point_count,
        keyframe.reprojection_threshold_px,
        f"homography-{digest}",
        keyframe.median_error_yards,
        keyframe.max_error_yards,
        keyframe.withheld_median_error_yards,
        keyframe.withheld_p95_error_yards,
        keyframe.withheld_point_count,
        keyframe.status,
        keyframe.reason,
        keyframe.withheld_errors_yards,
    )


def estimate_field_motion(
    previous_bgr: np.ndarray,
    current_bgr: np.ndarray,
    static_mask: np.ndarray | None = None,
    *,
    reprojection_threshold_px: float = 3.0,
) -> np.ndarray | None:
    """Estimate ``G(current <- previous)`` from static field pixels.

    The caller should provide a mask that excludes players, officials, score
    graphics and other moving overlays. Returning ``None`` is intentional when
    there is not enough spatial support; callers must then stop propagation.
    """

    if previous_bgr.ndim != 3 or current_bgr.ndim != 3 or previous_bgr.shape != current_bgr.shape:
        raise ValueError("motion frames must have matching BGR shapes")
    if not np.isfinite(reprojection_threshold_px) or reprojection_threshold_px <= 0:
        raise ValueError("motion reprojection threshold must be positive")
    previous_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    mask = None
    if static_mask is not None:
        if static_mask.shape[:2] != previous_gray.shape or static_mask.dtype != np.uint8:
            raise ValueError("static mask must be uint8 and match frame dimensions")
        mask = static_mask
    points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=800, qualityLevel=0.01, minDistance=12, mask=mask)
    if points is None or len(points) < 4:
        return None
    current_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None)
    if current_points is None or status is None:
        return None
    valid = status.reshape(-1).astype(bool)
    source = points.reshape(-1, 2)[valid]
    target = current_points.reshape(-1, 2)[valid]
    if len(source) < 4:
        return None
    motion, inliers = cv2.findHomography(source, target, cv2.RANSAC, reprojection_threshold_px)
    if motion is None or inliers is None or int(np.count_nonzero(inliers)) < 4 or abs(float(motion[2, 2])) <= 1e-12:
        return None
    motion = np.asarray(motion / motion[2, 2], dtype=np.float64)
    return motion if np.all(np.isfinite(motion)) else None


def static_field_mask(
    frame_shape: tuple[int, int] | tuple[int, int, int],
    excluded_boxes_xyxy: Sequence[Sequence[float]] = (),
    *,
    margin_px: int = 8,
) -> np.ndarray:
    """Build a uint8 mask for motion estimation, excluding dynamic boxes."""

    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0 or margin_px < 0:
        raise ValueError("frame shape and margin must be positive")
    mask = np.full((height, width), 255, dtype=np.uint8)
    for raw_box in excluded_boxes_xyxy:
        if len(raw_box) != 4 or not np.all(np.isfinite(np.asarray(raw_box, dtype=float))):
            raise ValueError("excluded boxes must contain four finite coordinates")
        x1, y1, x2, y2 = (int(round(float(item))) for item in raw_box)
        x1 = max(0, x1 - margin_px)
        y1 = max(0, y1 - margin_px)
        x2 = min(width, x2 + margin_px)
        y2 = min(height, y2 + margin_px)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
    return mask
