"""Robust image-to-yard field calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .schema import Observation


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
    return np.linalg.matrix_rank(centered, tol=1e-8) >= 2


@dataclass(frozen=True, slots=True)
class Homography:
    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    median_error_px: float
    max_error_px: float
    inlier_count: int
    point_count: int
    reprojection_threshold_px: float
    calibration_id: str

    @classmethod
    def fit(
        cls,
        image_points: Sequence[ImagePoint],
        field_points: Sequence[FieldPoint],
        reprojection_threshold_px: float = 3.0,
    ) -> "Homography":
        if len(image_points) != len(field_points) or len(image_points) < 4:
            raise ValueError("homography needs at least four paired landmarks")
        if reprojection_threshold_px <= 0:
            raise ValueError("reprojection threshold must be positive")
        image = np.asarray([(point.x_px, point.y_px) for point in image_points], dtype=np.float64)
        field = np.asarray([(point.x_yards, point.y_yards) for point in field_points], dtype=np.float64)
        if not _non_collinear(image) or not _non_collinear(field):
            raise ValueError("homography landmarks must be non-collinear")
        matrix, mask = cv2.findHomography(image, field, cv2.RANSAC, reprojection_threshold_px)
        if matrix is None or mask is None:
            raise ValueError("unable to fit homography")
        projected = cv2.perspectiveTransform(image.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        errors = np.linalg.norm(projected - field, axis=1)
        inlier_count = int(np.count_nonzero(mask.reshape(-1)))
        if inlier_count < 4:
            raise ValueError("homography has fewer than four inliers")
        canonical = np.asarray(matrix / matrix[2, 2], dtype=np.float64)
        digest = hashlib.sha256(canonical.tobytes()).hexdigest()[:16]
        matrix_tuple = tuple(tuple(float(value) for value in row) for row in canonical)
        return cls(
            matrix_tuple,
            float(np.median(errors)),
            float(np.max(errors)),
            inlier_count,
            len(image_points),
            float(reprojection_threshold_px),
            f"homography-{digest}",
        )

    @property
    def is_valid(self) -> bool:
        return self.inlier_count >= 4 and np.isfinite(self.median_error_px) and self.median_error_px <= self.reprojection_threshold_px * 2.0

    def project(self, image_point: ImagePoint) -> FieldPoint | None:
        if not self.is_valid:
            return None
        matrix = np.asarray(self.matrix, dtype=np.float64)
        point = np.asarray([[[image_point.x_px, image_point.y_px]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(point, matrix).reshape(2)
        if not np.all(np.isfinite(projected)):
            return None
        return FieldPoint(float(projected[0]), float(projected[1]))


def project_observation(observation: Observation, homography: Homography, source: str = "bottom_center") -> Observation:
    if source != "bottom_center":
        raise ValueError("only bottom_center projection is currently supported")
    x1, _, x2, y2 = observation.bbox_xyxy_px
    projected = homography.project(ImagePoint((x1 + x2) / 2.0, y2))
    if projected is None:
        return observation
    return replace(
        observation,
        field_xy_yards=(projected.x_yards, projected.y_yards),
        position_source=source,
        calibration_id=homography.calibration_id,
    )


def load_calibrations(path: str | Path) -> dict[str, Homography]:
    """Load either one shared calibration or a mapping keyed by shot id."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))

    def fit(config: dict) -> Homography:
        image_points = [ImagePoint(float(point[0]), float(point[1])) for point in config["image_points"]]
        field_points = [FieldPoint(float(point[0]), float(point[1])) for point in config["field_points"]]
        return Homography.fit(image_points, field_points, float(config.get("reprojection_threshold_px", 3.0)))

    if isinstance(value.get("shots"), dict):
        return {str(shot_id): fit(config) for shot_id, config in value["shots"].items()}
    return {"*": fit(value)}
