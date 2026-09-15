"""Robust image-to-yard field calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .schema import Observation
from .field import validate_field_point


@dataclass(frozen=True, slots=True)
class ImagePoint:
    x_px: float
    y_px: float


@dataclass(frozen=True, slots=True)
class FieldPoint:
    x_yards: float
    y_yards: float


def _non_collinear(points: np.ndarray) -> bool:
    if len(points) < 3:
        return False
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return np.linalg.matrix_rank(centered, tol=1e-8) >= 2 and bool(singular[1] / max(singular[0], 1e-12) > 1e-6)


@dataclass(frozen=True, slots=True)
class Homography:
    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    median_error_px: float
    max_error_px: float
    inlier_count: int
    point_count: int
    reprojection_threshold_px: float
    calibration_id: str
    median_error_yards: float = float("inf")
    max_error_yards: float = float("inf")
    withheld_median_error_yards: float | None = None
    withheld_p95_error_yards: float | None = None
    withheld_point_count: int = 0
    status: str = "unvalidated"
    reason: str | None = None
    withheld_errors_yards: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or abs(float(matrix[2, 2])) <= 1e-12:
            raise ValueError("homography matrix must be a finite, scaled 3x3 matrix")
        if self.inlier_count < 0 or self.point_count < 0 or self.inlier_count > self.point_count:
            raise ValueError("homography point counts are invalid")
        if not np.isfinite(self.reprojection_threshold_px) or self.reprojection_threshold_px <= 0:
            raise ValueError("homography reprojection threshold must be positive")
        if self.status not in {"valid", "unvalidated", "invalid", "partial"}:
            raise ValueError("unsupported homography status")
        for value in (self.median_error_px, self.max_error_px, self.median_error_yards, self.max_error_yards):
            if not np.isfinite(value) and value != float("inf"):
                raise ValueError("homography errors must be finite or positive infinity")
        if self.withheld_point_count < 0 or (self.withheld_errors_yards and len(self.withheld_errors_yards) != self.withheld_point_count):
            raise ValueError("withheld homography error count is invalid")
        if any(not np.isfinite(value) or value < 0 for value in self.withheld_errors_yards):
            raise ValueError("withheld homography errors must be finite and non-negative")
        for value in (self.withheld_median_error_yards, self.withheld_p95_error_yards):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("withheld homography errors must be finite and non-negative")

    @classmethod
    def fit(
        cls,
        image_points: Sequence[ImagePoint],
        field_points: Sequence[FieldPoint],
        reprojection_threshold_px: float = 3.0,
        *,
        withheld_image_points: Sequence[ImagePoint] | None = None,
        withheld_field_points: Sequence[FieldPoint] | None = None,
    ) -> "Homography":
        if len(image_points) != len(field_points) or len(image_points) < 4:
            raise ValueError("homography needs at least four paired landmarks")
        if not np.isfinite(reprojection_threshold_px) or reprojection_threshold_px <= 0:
            raise ValueError("reprojection threshold must be positive")
        image = np.asarray([(point.x_px, point.y_px) for point in image_points], dtype=np.float64)
        field = np.asarray([(point.x_yards, point.y_yards) for point in field_points], dtype=np.float64)
        if not np.all(np.isfinite(image)) or not np.all(np.isfinite(field)):
            raise ValueError("homography landmarks must be finite")
        for point in field:
            validate_field_point((float(point[0]), float(point[1])))
        if len(np.unique(image, axis=0)) < 4 or len(np.unique(field, axis=0)) < 4:
            raise ValueError("homography landmarks must contain four unique points")
        if not _non_collinear(image) or not _non_collinear(field):
            raise ValueError("homography landmarks must be non-collinear")
        # OpenCV measures the RANSAC residual in destination coordinates.  Fit
        # field -> image so the configured threshold really is in pixels, then
        # invert the result for the image -> field projection used downstream.
        field_to_image, mask = cv2.findHomography(field, image, cv2.RANSAC, reprojection_threshold_px)
        try:
            matrix = None if field_to_image is None else np.linalg.inv(field_to_image)
        except np.linalg.LinAlgError as error:
            raise ValueError("unable to invert fitted homography") from error
        if matrix is None or mask is None:
            raise ValueError("unable to fit homography")
        image_projected = cv2.perspectiveTransform(field.reshape(-1, 1, 2), field_to_image).reshape(-1, 2)
        field_projected = cv2.perspectiveTransform(image.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        image_errors = np.linalg.norm(image_projected - image, axis=1)
        field_errors = np.linalg.norm(field_projected - field, axis=1)
        inlier_count = int(np.count_nonzero(mask.reshape(-1)))
        if inlier_count < 4:
            raise ValueError("homography has fewer than four inliers")
        if abs(float(matrix[2, 2])) <= 1e-12 or not np.all(np.isfinite(matrix)):
            raise ValueError("fitted homography has an invalid scale")
        canonical = np.asarray(matrix / matrix[2, 2], dtype=np.float64)
        digest = hashlib.sha256(canonical.tobytes()).hexdigest()[:16]
        matrix_tuple = tuple(tuple(float(value) for value in row) for row in canonical)
        withheld_median: float | None = None
        withheld_p95: float | None = None
        withheld_count = 0
        withheld_errors_values: tuple[float, ...] = ()
        if (withheld_image_points is not None) != (withheld_field_points is not None):
            raise ValueError("withheld image and field landmarks must be supplied together")
        if withheld_image_points is not None and withheld_field_points is not None:
            if len(withheld_image_points) != len(withheld_field_points) or len(withheld_image_points) == 0:
                raise ValueError("withheld landmarks must contain paired points")
            withheld_image = np.asarray([(point.x_px, point.y_px) for point in withheld_image_points], dtype=np.float64)
            withheld_field = np.asarray([(point.x_yards, point.y_yards) for point in withheld_field_points], dtype=np.float64)
            if not np.all(np.isfinite(withheld_image)) or not np.all(np.isfinite(withheld_field)):
                raise ValueError("withheld landmarks must be finite")
            for point in withheld_field:
                validate_field_point((float(point[0]), float(point[1])))
            withheld_projected = cv2.perspectiveTransform(withheld_image.reshape(-1, 1, 2), canonical).reshape(-1, 2)
            withheld_errors = np.linalg.norm(withheld_projected - withheld_field, axis=1)
            withheld_count = len(withheld_errors)
            withheld_errors_values = tuple(float(error) for error in withheld_errors)
            withheld_median = float(np.median(withheld_errors))
            withheld_p95 = float(np.percentile(withheld_errors, 95))
        fit_valid = bool(np.isfinite(image_errors).all() and np.isfinite(field_errors).all())
        if not fit_valid:
            status, reason = "invalid", "nonfinite reprojection error"
        elif withheld_count:
            status = "valid" if withheld_median <= 1.0 and withheld_p95 <= 2.0 else "invalid"
            reason = None if status == "valid" else "withheld landmark error exceeds calibration gate"
        else:
            status, reason = "unvalidated", "no withheld landmarks supplied"
        return cls(
            matrix_tuple,
            float(np.median(image_errors)),
            float(np.max(image_errors)),
            inlier_count,
            len(image_points),
            float(reprojection_threshold_px),
            f"homography-{digest}",
            float(np.median(field_errors)),
            float(np.max(field_errors)),
            withheld_median,
            withheld_p95,
            withheld_count,
            status,
            reason,
            withheld_errors_values,
        )

    @property
    def fit_error_px(self) -> float:
        """Backward-compatible explicit name for the image-space fit error."""

        return self.median_error_px

    @property
    def identity_eligible(self) -> bool:
        """Whether independent withheld geometry supports identity use."""

        return self.is_valid and self.status == "valid" and self.withheld_point_count > 0 and self.withheld_p95_error_yards is not None

    @property
    def is_valid(self) -> bool:
        return (
            self.inlier_count >= 4
            and np.isfinite(self.median_error_px)
            and self.median_error_px <= self.reprojection_threshold_px * 2.0
            and np.isfinite(self.median_error_yards)
        )

    def project(self, image_point: ImagePoint) -> FieldPoint | None:
        if not self.is_valid:
            return None
        matrix = np.asarray(self.matrix, dtype=np.float64)
        homogeneous = matrix @ np.asarray([image_point.x_px, image_point.y_px, 1.0], dtype=np.float64)
        denominator = float(homogeneous[2])
        if abs(denominator) <= 1e-9:
            return None
        projected = homogeneous[:2] / denominator
        if not np.all(np.isfinite(projected)):
            return None
        if not (0.0 <= float(projected[0]) <= 120.0 and 0.0 <= float(projected[1]) <= 160.0 / 3.0):
            return None
        # Canonicalize harmless floating-point noise so exact grid landmarks
        # round-trip deterministically in JSON and fixture comparisons.
        return FieldPoint(float(round(float(projected[0]), 12)), float(round(float(projected[1]), 12)))


def project_observation(
    observation: Observation,
    homography: Homography,
    source: str = "bottom_center",
    *,
    calibration_status: str | None = None,
    calibration_reason: str | None = None,
    support_polygon_px: Sequence[tuple[float, float]] | None = None,
) -> Observation:
    if source != "bottom_center":
        raise ValueError("only bottom_center projection is currently supported")
    x1, _, x2, y2 = observation.bbox_xyxy_px
    if support_polygon_px:
        contact = ((x1 + x2) / 2.0, y2)
        polygon_values = np.asarray(support_polygon_px, dtype=np.float32)
        if polygon_values.ndim != 2 or polygon_values.shape[1] != 2 or len(polygon_values) < 3 or not np.all(np.isfinite(polygon_values)):
            raise ValueError("support polygon must contain at least three finite xy points")
        polygon = polygon_values.reshape(-1, 1, 2)
        if cv2.pointPolygonTest(polygon, contact, False) < 0:
            return replace(
                observation,
                calibration_id=homography.calibration_id,
                calibration_status="invalid",
                calibration_reason="contact point lies outside calibration support",
            )
    projected = homography.project(ImagePoint((x1 + x2) / 2.0, y2))
    if projected is None:
        return replace(
            observation,
            calibration_id=homography.calibration_id,
            calibration_status=calibration_status or homography.status,
            calibration_reason=calibration_reason or homography.reason or "projection_invalid",
        )
    return replace(
        observation,
        field_xy_yards=(projected.x_yards, projected.y_yards),
        position_source=source,
        calibration_id=homography.calibration_id,
        calibration_status=calibration_status or homography.status,
        calibration_reason=calibration_reason or homography.reason,
        position_uncertainty_yards=homography.withheld_p95_error_yards if homography.identity_eligible else None,
    )


def load_calibrations(path: str | Path, *, source_sha256: str | None = None) -> dict[str, Homography]:
    """Load either one shared calibration or a mapping keyed by shot id."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("calibration must be a JSON object")
    declared_source = value.get("source_sha256")
    if source_sha256 is not None and declared_source is not None and str(declared_source) != source_sha256:
        raise ValueError("calibration source sha256 does not match the input video")

    def fit(config: dict) -> Homography:
        if "matrix" in config and "image_points" not in config:
            try:
                matrix = tuple(tuple(float(item) for item in row) for row in config["matrix"])
                if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                    raise ValueError("matrix must be 3x3")
                matrix_array = np.asarray(matrix, dtype=float)
                if not np.all(np.isfinite(matrix_array)) or abs(float(matrix_array[2, 2])) <= 1e-12:
                    raise ValueError("matrix must be finite and non-singular")
                matrix_array /= matrix_array[2, 2]
                matrix = tuple(tuple(float(item) for item in row) for row in matrix_array)
                return Homography(
                    matrix,
                    float(config.get("fit_error_px", config.get("median_error_px", float("inf")))),
                    float(config.get("max_error_px", float("inf"))),
                    int(config.get("inlier_count", 0)),
                    int(config.get("point_count", 0)),
                    float(config.get("reprojection_threshold_px", 3.0)),
                    str(config.get("calibration_id", "serialized-calibration")),
                    float(config.get("median_error_yards", float("inf"))),
                    float(config.get("max_error_yards", float("inf"))),
                    None if config.get("withheld_median_error_yards") is None else float(config["withheld_median_error_yards"]),
                    None if config.get("withheld_p95_error_yards") is None else float(config["withheld_p95_error_yards"]),
                    int(config.get("withheld_point_count", 0)),
                    str(config.get("status", "unvalidated")),
                    config.get("reason") if isinstance(config.get("reason"), str) else None,
                    tuple(float(item) for item in config.get("withheld_errors_yards", ())),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid serialized homography") from error
        try:
            image_points = [ImagePoint(float(point[0]), float(point[1])) for point in config["image_points"]]
            field_points = [FieldPoint(float(point[0]), float(point[1])) for point in config["field_points"]]
            withheld_image = config.get("withheld_image_points")
            withheld_field = config.get("withheld_field_points")
            withheld_image_points = None if withheld_image is None else [ImagePoint(float(point[0]), float(point[1])) for point in withheld_image]
            withheld_field_points = None if withheld_field is None else [FieldPoint(float(point[0]), float(point[1])) for point in withheld_field]
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError("invalid calibration landmarks") from error
        return Homography.fit(
            image_points,
            field_points,
            float(config.get("reprojection_threshold_px", 3.0)),
            withheld_image_points=withheld_image_points,
            withheld_field_points=withheld_field_points,
        )

    if isinstance(value.get("shots"), dict):
        return {str(shot_id): fit(config) for shot_id, config in value["shots"].items()}
    return {"*": fit(value)}


def calibration_quality_report(calibrations: Mapping[str, Homography]) -> dict[str, object]:
    """Summarize withheld calibration quality without treating fit error as validation."""

    if not calibrations:
        return {"status": "not_provided", "shots": {}}
    per_shot: dict[str, dict[str, object]] = {}
    all_errors: list[float] = []
    for shot_id, homography in sorted(calibrations.items()):
        errors = list(homography.withheld_errors_yards)
        if not errors and homography.withheld_median_error_yards is not None:
            errors = [homography.withheld_median_error_yards]
        all_errors.extend(errors)
        per_shot[shot_id] = {
            "status": homography.status,
            "withheld_point_count": homography.withheld_point_count,
            "median_error_yards": homography.withheld_median_error_yards,
            "p95_error_yards": homography.withheld_p95_error_yards,
            "fit_error_px": homography.fit_error_px,
            "calibration_id": homography.calibration_id,
            "reason": homography.reason,
        }
    if not all_errors:
        status = "invalid" if any(item["status"] == "invalid" for item in per_shot.values()) else "unvalidated"
        return {"status": status, "shots": per_shot, "reason": "no withheld landmark errors"}
    median = float(np.median(np.asarray(all_errors, dtype=float)))
    p95 = float(np.percentile(np.asarray(all_errors, dtype=float), 95))
    valid = all(calibration.status == "valid" and calibration.identity_eligible for calibration in calibrations.values()) and median <= 1.0 and p95 <= 2.0
    return {"status": "valid" if valid else "invalid", "shots": per_shot, "withheld_point_count": len(all_errors), "median_error_yards": median, "p95_error_yards": p95, "gate": {"median_at_most_1_yard": median <= 1.0, "p95_at_most_2_yards": p95 <= 2.0}}
