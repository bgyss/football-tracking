"""Reviewed play-time alignment and cross-shot identity candidate scoring.

Shots are tracked independently, so nothing in image coordinates survives a cut.
This module converts media time into a shared play time anchored on a reviewed
event, and scores cross-shot tracklet pairs only from view-invariant evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .identity import TeamEvidence


class ReplayAlignmentError(ValueError):
    """Raised when an alignment cannot support a cross-shot join."""


@dataclass(frozen=True, slots=True)
class PlayAnchor:
    shot_id: str
    source_frame: int
    event: str
    source_pts: int | None = None
    play_time_s_value: float | None = None

    def __post_init__(self) -> None:
        if self.source_frame < 0:
            raise ValueError("anchor source_frame must be non-negative")
        if not self.event:
            raise ValueError("anchor event must be named")
        if self.source_pts is not None and self.source_pts < 0:
            raise ValueError("anchor source_pts must be non-negative")
        if self.play_time_s_value is not None and not np.isfinite(self.play_time_s_value):
            raise ValueError("anchor play_time_s must be finite")
        if self.play_time_s_value is not None and self.source_pts is None:
            raise ValueError("anchor play_time_s requires source_pts")


@dataclass(frozen=True, slots=True)
class PlayAlignment:
    play_id: str
    anchors: tuple[PlayAnchor, ...]
    time_map: "PlayTimeMap | None" = None

    def shots(self) -> tuple[str, ...]:
        return tuple(sorted(anchor.shot_id for anchor in self.anchors))

    def play_time_s(self, shot_id: str, frame_index: int, fps: float) -> float | None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        for anchor in self.anchors:
            if anchor.shot_id == shot_id:
                return (frame_index - anchor.source_frame) / fps
        return None

    def play_time_at_pts(self, shot_id: str, pts: int, time_base: tuple[int, int], fps: float | None = None, frame_index: int | None = None) -> float | None:
        """Map source PTS to play time, with a legacy frame/fps fallback."""

        if self.time_map is not None:
            mapped = self.time_map.at(shot_id, pts)
            if mapped is not None:
                return mapped
        if fps is not None and frame_index is not None:
            return self.play_time_s(shot_id, frame_index, fps)
        return None


@dataclass(frozen=True, slots=True)
class PlayTimeCorrespondence:
    shot_id: str
    pts: int
    play_time_s: float
    event: str

    def __post_init__(self) -> None:
        if self.pts < 0 or not np.isfinite(self.play_time_s) or not self.event:
            raise ValueError("invalid play-time correspondence")


class PlayTimeMap:
    """Piecewise-linear, source-PTS to common play-time mapping.

    Mapping never extrapolates outside reviewed correspondences. This prevents
    a freeze, edit or slow-motion section from fabricating overlap.
    """

    def __init__(self, correspondences: Sequence[PlayTimeCorrespondence], *, max_residual_ms: float = 50.0) -> None:
        if not np.isfinite(max_residual_ms) or max_residual_ms <= 0:
            raise ValueError("max_residual_ms must be positive")
        by_shot: dict[str, tuple[PlayTimeCorrespondence, ...]] = {}
        for shot_id in sorted({item.shot_id for item in correspondences}):
            values = tuple(sorted((item for item in correspondences if item.shot_id == shot_id), key=lambda item: item.pts))
            if len(values) < 2:
                raise ReplayAlignmentError(f"play-time map needs two correspondences for {shot_id}")
            events = [item.event for item in values]
            if len(set(events)) != len(events):
                raise ReplayAlignmentError(f"play-time map has duplicate event labels for {shot_id}")
            if any(left.pts == right.pts or right.play_time_s <= left.play_time_s for left, right in zip(values, values[1:])):
                raise ReplayAlignmentError(f"play-time map must be strictly increasing for {shot_id}")
            by_shot[shot_id] = values
        if not by_shot:
            raise ReplayAlignmentError("play-time map has no correspondences")
        self._correspondences = by_shot
        self.max_residual_ms = float(max_residual_ms)

    def at(self, shot_id: str, pts: int) -> float | None:
        if pts < 0:
            raise ValueError("pts must be non-negative")
        values = self._correspondences.get(str(shot_id), ())
        if not values or pts < values[0].pts or pts > values[-1].pts:
            return None
        for left, right in zip(values, values[1:]):
            if left.pts <= pts <= right.pts:
                ratio = (pts - left.pts) / float(right.pts - left.pts)
                return float(left.play_time_s + ratio * (right.play_time_s - left.play_time_s))
        return None

    def shots(self) -> tuple[str, ...]:
        return tuple(sorted(self._correspondences))

    def validate(self, checks: Sequence[PlayTimeCorrespondence]) -> dict[str, object]:
        """Evaluate held-out reviewed events without refitting the map."""

        residuals: list[float] = []
        missing: list[dict[str, object]] = []
        for check in checks:
            predicted = self.at(check.shot_id, check.pts)
            if predicted is None:
                missing.append({"shot_id": check.shot_id, "pts": check.pts, "event": check.event})
            else:
                residuals.append(abs(predicted - check.play_time_s) * 1000.0)
        maximum = max(residuals, default=float("inf") if missing else 0.0)
        return {"status": "valid" if not missing and maximum <= self.max_residual_ms else "invalid", "max_residual_ms": maximum, "residuals_ms": residuals, "missing": missing, "gate": maximum <= self.max_residual_ms and not missing}

    def to_dict(self) -> dict[str, object]:
        return {
            "max_residual_ms": self.max_residual_ms,
            "shots": {
                shot_id: [{"pts": item.pts, "play_time_s": item.play_time_s, "event": item.event} for item in values]
                for shot_id, values in sorted(self._correspondences.items())
            },
        }


def load_play_alignment(path: str | Path) -> PlayAlignment:
    """Load reviewed snap anchors. Model-generated alignments are refused."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayAlignmentError(f"unable to read alignment: {error}") from error
    if not isinstance(value, dict) or value.get("reviewed") is not True:
        raise ReplayAlignmentError("alignment must be explicitly marked reviewed: true")
    play_id = str(value.get("play_id") or "")
    if not play_id:
        raise ReplayAlignmentError("alignment must name a play_id")
    raw_anchors = value.get("anchors")
    if not isinstance(raw_anchors, list) or len(raw_anchors) < 2:
        raise ReplayAlignmentError("alignment needs a reviewed anchor for at least two shots")
    anchors: list[PlayAnchor] = []
    seen: set[str] = set()
    for raw in raw_anchors:
        if not isinstance(raw, dict) or "shot_id" not in raw or "source_frame" not in raw:
            raise ReplayAlignmentError(f"invalid anchor: {raw!r}")
        shot_id = str(raw["shot_id"])
        if shot_id in seen:
            raise ReplayAlignmentError(f"duplicate anchor for {shot_id}")
        seen.add(shot_id)
        try:
            anchors.append(PlayAnchor(shot_id, int(raw["source_frame"]), str(raw.get("event", "snap")), None if raw.get("source_pts") is None else int(raw["source_pts"]), None if raw.get("play_time_s") is None else float(raw["play_time_s"])))
        except (TypeError, ValueError) as error:
            raise ReplayAlignmentError(f"invalid anchor for {shot_id}: {error}") from error
    ordered = tuple(sorted(anchors, key=lambda anchor: anchor.shot_id))
    correspondences = [PlayTimeCorrespondence(anchor.shot_id, anchor.source_pts, anchor.play_time_s_value, anchor.event) for anchor in ordered if anchor.source_pts is not None and anchor.play_time_s_value is not None]
    raw_correspondences = value.get("correspondences", [])
    if raw_correspondences:
        if not isinstance(raw_correspondences, list):
            raise ReplayAlignmentError("correspondences must be a list")
        for raw in raw_correspondences:
            if not isinstance(raw, dict):
                raise ReplayAlignmentError(f"invalid correspondence: {raw!r}")
            try:
                correspondences.append(PlayTimeCorrespondence(str(raw["shot_id"]), int(raw["source_pts"]), float(raw["play_time_s"]), str(raw["event"])))
            except (KeyError, TypeError, ValueError) as error:
                raise ReplayAlignmentError(f"invalid correspondence: {raw!r}") from error
    time_map = None
    if correspondences and all(sum(item.shot_id == shot_id for item in correspondences) >= 2 for shot_id in {item.shot_id for item in correspondences}):
        try:
            time_map = PlayTimeMap(correspondences, max_residual_ms=float(value.get("max_residual_ms", 50.0)))
        except ReplayAlignmentError:
            raise
    return PlayAlignment(play_id, ordered, time_map)


@dataclass(frozen=True, slots=True)
class FieldTrack:
    """A tracklet resampled into (play_time_s, field_x_yards, field_y_yards)."""

    tracklet_id: str
    shot_id: str
    samples: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if any(len(sample) != 3 for sample in self.samples):
            raise ValueError("samples must be (play_time_s, field_x_yards, field_y_yards)")


def _paired_samples(
    left: FieldTrack,
    right: FieldTrack,
    tolerance_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair samples that share a play time within tolerance.

    Each right-hand sample is consumable at most once. Left samples are walked
    in order, and the nearest unmatched right sample within tolerance is claimed.
    This ensures a one-to-one correspondence in case of occlusion gaps or
    differently-sampled tracklets.
    """

    if not left.samples or not right.samples:
        return np.empty((0, 2)), np.empty((0, 2))
    right_times = np.asarray([sample[0] for sample in right.samples], dtype=float)
    right_points = np.asarray([(sample[1], sample[2]) for sample in right.samples], dtype=float)
    left_pairs: list[tuple[float, float]] = []
    right_pairs: list[tuple[float, float]] = []
    used: set[int] = set()
    for time_s, x, y in left.samples:
        offsets = np.abs(right_times - time_s)
        if used:
            offsets = offsets.copy()
            offsets[list(used)] = np.inf
        index = int(np.argmin(offsets))
        if offsets[index] <= tolerance_s:
            used.add(index)
            left_pairs.append((x, y))
            right_pairs.append(tuple(right_points[index]))
    return np.asarray(left_pairs, dtype=float), np.asarray(right_pairs, dtype=float)


def _shape_agreement(left_points: np.ndarray, right_points: np.ndarray) -> float:
    """Cosine agreement of net displacement, rescaled to [0, 1]."""

    if len(left_points) < 2:
        return 0.0
    left_delta = left_points[-1] - left_points[0]
    right_delta = right_points[-1] - right_points[0]
    left_norm = float(np.linalg.norm(left_delta))
    right_norm = float(np.linalg.norm(right_delta))
    if left_norm < 1e-6 or right_norm < 1e-6:
        # Two stationary players agree on shape but carry no directional evidence.
        return 0.5
    cosine = float(np.dot(left_delta, right_delta) / (left_norm * right_norm))
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def cross_shot_candidate_scores(
    left: Sequence[FieldTrack],
    right: Sequence[FieldTrack],
    teams: Mapping[str, TeamEvidence],
    *,
    max_field_distance_yards: float = 6.0,
    min_overlap_samples: int = 5,
    sample_tolerance_s: float = 0.05,
    min_overlap_duration_s: float = 0.4,
) -> dict[tuple[str, str], float]:
    """Score cross-shot pairs from view-invariant evidence only.

    A pair that fails a hard constraint is omitted entirely rather than scored
    low, so ``match_tracklets`` reports ``insufficient_evidence`` instead of a
    weak ``same``.
    """

    if max_field_distance_yards <= 0 or min_overlap_samples < 2 or sample_tolerance_s <= 0 or min_overlap_duration_s < 0:
        raise ValueError("invalid cross-shot scoring constraints")
    scores: dict[tuple[str, str], float] = {}
    for left_track in sorted(left, key=lambda track: track.tracklet_id):
        left_team = teams.get(left_track.tracklet_id)
        if left_team is None or not left_team.eligible:
            continue
        for right_track in sorted(right, key=lambda track: track.tracklet_id):
            right_team = teams.get(right_track.tracklet_id)
            if right_team is None or not right_team.eligible:
                continue
            if left_team.team != right_team.team:
                continue
            left_points, right_points = _paired_samples(left_track, right_track, sample_tolerance_s)
            if len(left_points) < min_overlap_samples:
                continue
            paired_times = []
            right_times = np.asarray([sample[0] for sample in right_track.samples], dtype=float)
            used_right: set[int] = set()
            for time_s, _, _ in left_track.samples:
                offsets = np.abs(right_times - time_s)
                if used_right:
                    offsets = offsets.copy()
                    offsets[list(used_right)] = np.inf
                index = int(np.argmin(offsets))
                if offsets[index] <= sample_tolerance_s:
                    used_right.add(index)
                    paired_times.append(float(time_s))
            if paired_times and max(paired_times) - min(paired_times) < min_overlap_duration_s:
                continue
            distances = np.linalg.norm(left_points - right_points, axis=1)
            median_distance = float(np.median(distances))
            p90_distance = float(np.percentile(distances, 90))
            if p90_distance > max_field_distance_yards:
                continue
            position = 1.0 - (median_distance / max_field_distance_yards)
            shape = _shape_agreement(left_points, right_points)
            team_confidence = float(left_team.score * right_team.score)
            score = 0.55 * position + 0.30 * shape + 0.15 * team_confidence
            scores[(left_track.tracklet_id, right_track.tracklet_id)] = max(0.0, min(1.0, score))
    return scores
