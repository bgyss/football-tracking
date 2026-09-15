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

    def __post_init__(self) -> None:
        if self.source_frame < 0:
            raise ValueError("anchor source_frame must be non-negative")
        if not self.event:
            raise ValueError("anchor event must be named")


@dataclass(frozen=True, slots=True)
class PlayAlignment:
    play_id: str
    anchors: tuple[PlayAnchor, ...]

    def shots(self) -> tuple[str, ...]:
        return tuple(sorted(anchor.shot_id for anchor in self.anchors))

    def play_time_s(self, shot_id: str, frame_index: int, fps: float) -> float | None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        for anchor in self.anchors:
            if anchor.shot_id == shot_id:
                return (frame_index - anchor.source_frame) / fps
        return None


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
            anchors.append(PlayAnchor(shot_id, int(raw["source_frame"]), str(raw.get("event", "snap"))))
        except (TypeError, ValueError) as error:
            raise ReplayAlignmentError(f"invalid anchor for {shot_id}: {error}") from error
    return PlayAlignment(play_id, tuple(sorted(anchors, key=lambda anchor: anchor.shot_id)))


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
) -> dict[tuple[str, str], float]:
    """Score cross-shot pairs from view-invariant evidence only.

    A pair that fails a hard constraint is omitted entirely rather than scored
    low, so ``match_tracklets`` reports ``insufficient_evidence`` instead of a
    weak ``same``.
    """

    if max_field_distance_yards <= 0 or min_overlap_samples < 2:
        raise ValueError("invalid cross-shot scoring constraints")
    scores: dict[tuple[str, str], float] = {}
    for left_track in sorted(left, key=lambda track: track.tracklet_id):
        left_team = teams.get(left_track.tracklet_id)
        if left_team is None or left_team.team == "unknown":
            continue
        for right_track in sorted(right, key=lambda track: track.tracklet_id):
            right_team = teams.get(right_track.tracklet_id)
            if right_team is None or right_team.team == "unknown":
                continue
            if left_team.team != right_team.team:
                continue
            left_points, right_points = _paired_samples(left_track, right_track, sample_tolerance_s)
            if len(left_points) < min_overlap_samples:
                continue
            distances = np.linalg.norm(left_points - right_points, axis=1)
            mean_distance = float(np.mean(distances))
            if mean_distance > max_field_distance_yards:
                continue
            position = 1.0 - (mean_distance / max_field_distance_yards)
            shape = _shape_agreement(left_points, right_points)
            team_confidence = float(left_team.score * right_team.score)
            score = 0.55 * position + 0.30 * shape + 0.15 * team_confidence
            scores[(left_track.tracklet_id, right_track.tracklet_id)] = max(0.0, min(1.0, score))
    return scores
