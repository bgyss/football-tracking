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
                            "inlier_count": estimate.homography.inlier_count,
                            "point_count": estimate.homography.point_count,
                            "reprojection_threshold_px": estimate.homography.reprojection_threshold_px,
                            "calibration_id": estimate.homography.calibration_id,
                            "status": estimate.homography.status,
                        },
                    }
                    for estimate in values
                ]
            }
        return {"schema_version": 2, "shots": shots}


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
    withheld_image = None if withheld_image_raw is None else [ImagePoint(float(point[0]), float(point[1])) for point in withheld_image_raw]  # type: ignore[index]
    withheld_field = None if withheld_field_raw is None else [FieldPoint(float(point[0]), float(point[1])) for point in withheld_field_raw]  # type: ignore[index]
    return image, field, withheld_image, withheld_field


def load_calibration_timeline(path: str | Path) -> CalibrationTimeline:
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
                        matrix, float(serialized.get("fit_error_px", serialized.get("median_error_px", float("inf")))), float(serialized.get("max_error_px", float("inf"))), int(serialized["inlier_count"]), int(serialized["point_count"]), float(serialized.get("reprojection_threshold_px", 3.0)), str(serialized.get("calibration_id", f"serialized-{shot_id}-{index}")), float(serialized.get("median_error_yards", float("inf"))), float(serialized.get("max_error_yards", float("inf"))), None if serialized.get("withheld_median_error_yards") is None else float(serialized["withheld_median_error_yards"]), None if serialized.get("withheld_p95_error_yards") is None else float(serialized["withheld_p95_error_yards"]), int(serialized.get("withheld_point_count", 0)), str(serialized.get("status", config.get("status", "unvalidated"))), serialized.get("reason") if isinstance(serialized.get("reason"), str) else None,
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
            pts_start = int(config.get("pts_start", 0))
            pts_end_value = config.get("pts_end")
            pts_end = None if pts_end_value is None else int(pts_end_value)
            keyframe_id = str(config.get("keyframe_id", f"{shot_id}:k{index}"))
            raw_polygon = config.get("support_polygon_px", ())
            try:
                support_polygon = tuple((float(point[0]), float(point[1])) for point in raw_polygon)  # type: ignore[index]
            except (TypeError, ValueError, IndexError) as error:
                raise CalibrationTimelineError(f"invalid support polygon {shot_id}:{index}") from error
            if support_polygon and len(support_polygon) < 3:
                raise CalibrationTimelineError("support polygon needs at least three points")
            # A timeline fit without withheld points remains mathematically
            # projectable but is not a promoted calibration result.
            status = str(config.get("status", homography.status))
            if status not in {"valid", "unvalidated", "invalid", "partial"}:
                raise CalibrationTimelineError(f"unsupported calibration status {status!r}")
            calibration_id = hashlib.sha256(f"{shot_id}:{keyframe_id}:{homography.calibration_id}".encode()).hexdigest()[:16]
            estimates.append(CalibrationEstimate(str(shot_id), calibration_id, homography, pts_start, pts_end, (keyframe_id,), status, config.get("reason") if isinstance(config.get("reason"), str) else homography.reason, support_polygon))
        result[str(shot_id)] = estimates
    return CalibrationTimeline(result)


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
    if reprojection_threshold_px <= 0:
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
