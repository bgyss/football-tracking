"""Stable data contracts shared by the tracking pipeline and its exporters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    return value


@dataclass(frozen=True, slots=True)
class Observation:
    """One detector/tracker observation in original image coordinates."""

    run_id: str
    shot_id: str
    frame_index: int
    pts: int
    time_base: tuple[int, int]
    tracklet_id: str
    player_id: str | None
    bbox_xyxy_px: tuple[float, float, float, float]
    detection_score: float
    team: str
    team_score: float
    jersey_number: int | None
    field_xy_yards: tuple[float, float] | None
    position_source: str | None
    calibration_id: str | None
    identity_version: int
    play_id: str | None = None
    calibration_status: str | None = None
    calibration_reason: str | None = None
    position_uncertainty_yards: float | None = None
    play_time_s: float | None = None
    time_map_id: str | None = None
    source_tracklet_id: str | None = None

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy_px
        if any(not math.isfinite(float(value)) for value in self.bbox_xyxy_px):
            raise ValueError("bbox coordinates must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive width and height")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.pts < 0:
            raise ValueError("pts must be non-negative")
        if len(self.time_base) != 2 or self.time_base[0] <= 0 or self.time_base[1] <= 0:
            raise ValueError("time_base must contain positive numerator and denominator")
        if not 0.0 <= self.detection_score <= 1.0:
            raise ValueError("detection_score must be between 0 and 1")
        if not 0.0 <= self.team_score <= 1.0:
            raise ValueError("team_score must be between 0 and 1")
        if self.jersey_number is not None and not 0 <= self.jersey_number <= 99:
            raise ValueError("jersey_number must be between 0 and 99")
        if self.field_xy_yards is not None and len(self.field_xy_yards) != 2:
            raise ValueError("field_xy_yards must contain x and y")
        if self.field_xy_yards is not None and any(not math.isfinite(float(value)) for value in self.field_xy_yards):
            raise ValueError("field_xy_yards must be finite")
        if self.identity_version < 0:
            raise ValueError("identity_version must be non-negative")
        if self.position_uncertainty_yards is not None and self.position_uncertainty_yards < 0:
            raise ValueError("position_uncertainty_yards must be non-negative")
        if self.position_uncertainty_yards is not None and not math.isfinite(float(self.position_uncertainty_yards)):
            raise ValueError("position_uncertainty_yards must be finite")
        if self.play_time_s is not None and not math.isfinite(float(self.play_time_s)):
            raise ValueError("play_time_s must be finite")
        if self.calibration_status is not None and self.calibration_status not in {"not_provided", "unvalidated", "valid", "partial", "invalid"}:
            raise ValueError("unsupported calibration_status")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Reproducibility record for one batch run."""

    run_id: str
    input_path: str
    input_sha256: str
    input_duration_s: float
    source_fps: float
    frame_count: int
    detector: str
    detector_version: str
    tracker: str
    tracker_version: str
    config_hash: str
    device: str
    status: str
    timings: dict[str, float] = field(default_factory=dict)
    api_usage: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_duration_s < 0:
            raise ValueError("input_duration_s must be non-negative")
        if self.source_fps <= 0:
            raise ValueError("source_fps must be positive")
        if self.frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        if self.status not in {"complete", "complete_with_unresolved", "incomplete", "failed"}:
            raise ValueError("unsupported run status")

    def to_dict(self) -> dict[str, Any]:
        value = _json_safe(asdict(self))
        return {key: value[key] for key in sorted(value)}
