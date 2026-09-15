"""Reviewed play-time alignment and cross-shot identity candidate scoring.

Shots are tracked independently, so nothing in image coordinates survives a cut.
This module converts media time into a shared play time anchored on a reviewed
event, and scores cross-shot tracklet pairs only from view-invariant evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
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
        if not isinstance(self.shot_id, str) or not self.shot_id.strip():
            raise ValueError("anchor shot_id must be non-empty")
        if not isinstance(self.source_frame, int) or isinstance(self.source_frame, bool) or self.source_frame < 0:
            raise ValueError("anchor source_frame must be non-negative")
        if not isinstance(self.event, str) or not self.event.strip():
            raise ValueError("anchor event must be named")
        if self.source_pts is not None and (not isinstance(self.source_pts, int) or isinstance(self.source_pts, bool) or self.source_pts < 0):
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
    shot_ranges: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    timing_report: Mapping[str, object] | None = None
    source_hash_validated: bool = True
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.play_id or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.play_id) is None:
            raise ReplayAlignmentError("alignment play_id must be a safe identifier")
        anchor_shots = [anchor.shot_id for anchor in self.anchors]
        if len(set(anchor_shots)) != len(anchor_shots):
            raise ReplayAlignmentError("alignment contains duplicate shot anchors")
        for shot_id, interval in self.shot_ranges.items():
            if not isinstance(shot_id, str) or len(interval) != 2 or any(not isinstance(value, int) or isinstance(value, bool) for value in interval) or interval[0] < 0 or interval[1] <= interval[0]:
                raise ReplayAlignmentError(f"invalid shot range for {shot_id}")
            if shot_id not in anchor_shots:
                raise ReplayAlignmentError(f"shot range has no anchor for {shot_id}")

    def shots(self) -> tuple[str, ...]:
        return tuple(sorted(set(anchor.shot_id for anchor in self.anchors) | set(self.shot_ranges)))

    @property
    def timing_eligible(self) -> bool:
        return self.source_hash_validated and self.time_map is not None and bool(self.timing_report and self.timing_report.get("gate") is True)

    def play_time_s(self, shot_id: str, frame_index: int, fps: float) -> float | None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        for anchor in self.anchors:
            if anchor.shot_id == shot_id:
                return (frame_index - anchor.source_frame) / fps
        return None

    def play_time_at_pts(self, shot_id: str, pts: int, time_base: tuple[int, int], fps: float | None = None, frame_index: int | None = None) -> float | None:
        """Map source PTS to play time, with a legacy frame/fps fallback."""

        if pts < 0 or len(time_base) != 2 or time_base[0] <= 0 or time_base[1] <= 0:
            raise ValueError("invalid source PTS or time base")
        if self.time_map is not None:
            # A fitted map is deliberately bounded by reviewed
            # correspondences. Do not fall back to an affine anchor outside
            # that support: doing so would silently extrapolate through edits,
            # freezes, or gaps.
            return self.time_map.at(shot_id, pts)
        for anchor in self.anchors:
            if anchor.shot_id == shot_id and anchor.source_pts is not None:
                anchor_time = anchor.play_time_s_value or 0.0
                return anchor_time + (pts - anchor.source_pts) * time_base[0] / time_base[1]
        if fps is not None and frame_index is not None:
            return self.play_time_s(shot_id, frame_index, fps)
        return None


@dataclass(frozen=True, slots=True)
class PlayAlignmentSet:
    """A deterministic collection of reviewed play alignments."""

    plays: tuple[PlayAlignment, ...]

    def __post_init__(self) -> None:
        ids = [play.play_id for play in self.plays]
        if len(set(ids)) != len(ids):
            raise ReplayAlignmentError("alignment set contains duplicate play_id values")

    def by_id(self, play_id: str) -> PlayAlignment | None:
        return next((play for play in self.plays if play.play_id == play_id), None)

    def to_dict(self) -> dict[str, object]:
        source_hashes = {play.source_sha256 for play in self.plays if play.source_sha256}
        value: dict[str, object] = {"reviewed": True, "plays": [{"play_id": play.play_id, "source_sha256": play.source_sha256, "anchors": [{"shot_id": anchor.shot_id, "source_frame": anchor.source_frame, "source_pts": anchor.source_pts, "play_time_s": anchor.play_time_s_value, "event": anchor.event} for anchor in play.anchors], "shot_ranges": {shot_id: list(interval) for shot_id, interval in sorted(play.shot_ranges.items())}, "correspondences": [{"shot_id": shot_id, **item} for shot_id, values in (play.time_map.to_dict().get("shots", {}) if play.time_map is not None else {}).items() for item in values], "max_residual_ms": play.time_map.max_residual_ms if play.time_map is not None else 50.0, "timing_report": play.timing_report} for play in self.plays]}
        if len(source_hashes) == 1:
            value["source_sha256"] = next(iter(source_hashes))
        return value

@dataclass(frozen=True, slots=True)
class PlayTimeCorrespondence:
    shot_id: str
    pts: int
    play_time_s: float
    event: str

    def __post_init__(self) -> None:
        if not isinstance(self.shot_id, str) or not self.shot_id.strip() or not isinstance(self.pts, int) or isinstance(self.pts, bool) or self.pts < 0 or not np.isfinite(self.play_time_s) or not isinstance(self.event, str) or not self.event.strip():
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
        event_sets = {shot_id: {item.event for item in values} for shot_id, values in by_shot.items()}
        expected_events = next(iter(event_sets.values()))
        if any(events != expected_events for events in event_sets.values()):
            raise ReplayAlignmentError("play-time map must use the same reviewed events in every shot")
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


def _parse_play_alignment_value(value: object, *, source_sha256: str | None = None) -> PlayAlignment:
    if not isinstance(value, dict) or value.get("reviewed") is not True:
        raise ReplayAlignmentError("alignment must be explicitly marked reviewed: true")
    declared_source = value.get("source_sha256")
    source_hash_validated = source_sha256 is None or (declared_source is not None and str(declared_source) == source_sha256)
    if source_sha256 is not None and declared_source is not None and str(declared_source) != source_sha256:
        raise ReplayAlignmentError("alignment source sha256 does not match the input video")
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
    timing_report: dict[str, object] | None = None
    if correspondences and all(sum(item.shot_id == shot_id for item in correspondences) >= 2 for shot_id in {item.shot_id for item in correspondences}):
        try:
            time_map = PlayTimeMap(correspondences, max_residual_ms=float(value.get("max_residual_ms", 50.0)))
            raw_checks = value.get("validation_correspondences", [])
            checks: list[PlayTimeCorrespondence] = []
            if raw_checks:
                if not isinstance(raw_checks, list):
                    raise ReplayAlignmentError("validation_correspondences must be a list")
                for raw in raw_checks:
                    if not isinstance(raw, dict):
                        raise ReplayAlignmentError(f"invalid validation correspondence: {raw!r}")
                    try:
                        checks.append(PlayTimeCorrespondence(str(raw["shot_id"]), int(raw["source_pts"]), float(raw["play_time_s"]), str(raw["event"])))
                    except (KeyError, TypeError, ValueError) as error:
                        raise ReplayAlignmentError(f"invalid validation correspondence: {raw!r}") from error
            if checks and {check.shot_id for check in checks} != set(time_map.shots()):
                raise ReplayAlignmentError("validation correspondence must cover every mapped shot")
            if len({(check.shot_id, check.event) for check in checks}) != len(checks):
                raise ReplayAlignmentError("validation correspondence has duplicate event labels")
            check_events = {shot_id: {check.event for check in checks if check.shot_id == shot_id} for shot_id in time_map.shots()}
            expected_check_events = next(iter(check_events.values()), set())
            if any(events != expected_check_events for events in check_events.values()):
                raise ReplayAlignmentError("validation correspondence must use the same reviewed events in every shot")
            fitted_events = {(item.shot_id, item.event) for item in correspondences}
            if any((check.shot_id, check.event) in fitted_events for check in checks):
                raise ReplayAlignmentError("validation correspondence must use events held out from the fit")
            check_points = [(check.shot_id, check.pts) for check in checks]
            if len(set(check_points)) != len(check_points):
                raise ReplayAlignmentError("validation correspondence has duplicate PTS values")
            timing_report = dict(time_map.validate(checks)) if checks else {"status": "unvalidated", "gate": False, "reason": "no held-out validation correspondence"}
        except ReplayAlignmentError:
            raise
    raw_ranges = value.get("shot_ranges", {})
    if raw_ranges is None:
        raw_ranges = {}
    if not isinstance(raw_ranges, dict):
        raise ReplayAlignmentError("shot_ranges must be an object")
    shot_ranges: dict[str, tuple[int, int]] = {}
    for raw_shot, raw_interval in raw_ranges.items():
        try:
            start, end = int(raw_interval[0]), int(raw_interval[1])
        except (TypeError, ValueError, IndexError) as error:
            raise ReplayAlignmentError(f"invalid shot range for {raw_shot}") from error
        if start < 0 or end <= start:
            raise ReplayAlignmentError(f"invalid shot range for {raw_shot}")
        shot_ranges[str(raw_shot)] = (start, end)
    if set(shot_ranges) - set(anchor.shot_id for anchor in ordered):
        missing_anchor_shots = sorted(set(shot_ranges) - set(anchor.shot_id for anchor in ordered))
        raise ReplayAlignmentError(f"shot ranges have no anchor: {missing_anchor_shots}")
    for anchor in ordered:
        interval = shot_ranges.get(anchor.shot_id)
        if interval is not None and not interval[0] <= anchor.source_frame < interval[1]:
            raise ReplayAlignmentError(f"anchor {anchor.shot_id}:{anchor.source_frame} lies outside its declared shot range")
    anchored_shots = {anchor.shot_id for anchor in ordered}
    correspondence_shots = {item.shot_id for item in correspondences}
    if correspondence_shots - anchored_shots:
        raise ReplayAlignmentError(f"correspondence has no anchor for shot(s): {sorted(correspondence_shots - anchored_shots)}")
    if time_map is not None:
        event_sets = {shot_id: {item.event for item in correspondences if item.shot_id == shot_id} for shot_id in time_map.shots()}
        expected_events = next(iter(event_sets.values()))
        if any(events != expected_events for events in event_sets.values()):
            raise ReplayAlignmentError("correspondences must use the same reviewed events in every shot")
    return PlayAlignment(play_id, ordered, time_map, shot_ranges, timing_report, source_hash_validated, None if declared_source is None else str(declared_source))


def load_play_alignment(path: str | Path, *, source_sha256: str | None = None) -> PlayAlignment:
    """Load one reviewed play alignment. Model-generated alignments are refused."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayAlignmentError(f"unable to read alignment: {error}") from error
    if isinstance(value, dict) and isinstance(value.get("plays"), list):
        if len(value["plays"]) != 1:
            raise ReplayAlignmentError("alignment contains multiple plays; use load_play_alignments")
        nested = dict(value["plays"][0]) if isinstance(value["plays"][0], dict) else value["plays"][0]
        if isinstance(nested, dict):
            nested.setdefault("reviewed", value.get("reviewed"))
            if "source_sha256" not in nested and value.get("source_sha256") is not None:
                nested["source_sha256"] = value["source_sha256"]
        value = nested
    return _parse_play_alignment_value(value, source_sha256=source_sha256)


def load_play_alignments(path: str | Path, *, source_sha256: str | None = None) -> PlayAlignmentSet:
    """Load a reviewed multi-play alignment manifest."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayAlignmentError(f"unable to read alignments: {error}") from error
    if not isinstance(value, dict) or value.get("reviewed") is not True:
        raise ReplayAlignmentError("alignment set must be explicitly marked reviewed: true")
    raw_plays = value.get("plays")
    if raw_plays == []:
        raise ReplayAlignmentError("alignment set contains no plays")
    if not isinstance(raw_plays, list):
        return PlayAlignmentSet((load_play_alignment(path, source_sha256=source_sha256),))
    plays: list[PlayAlignment] = []
    for raw_play in raw_plays:
        if not isinstance(raw_play, dict):
            raise ReplayAlignmentError(f"invalid play alignment: {raw_play!r}")
        nested = dict(raw_play)
        nested["reviewed"] = True
        if "source_sha256" not in nested and value.get("source_sha256") is not None:
            nested["source_sha256"] = value["source_sha256"]
        plays.append(_parse_play_alignment_value(nested, source_sha256=source_sha256))
    return PlayAlignmentSet(tuple(sorted(plays, key=lambda play: play.play_id)))


@dataclass(frozen=True, slots=True)
class FieldTrack:
    """A tracklet resampled into (play_time_s, field_x_yards, field_y_yards)."""

    tracklet_id: str
    shot_id: str
    samples: tuple[tuple[float, float, float], ...] = ()
    uncertainty_yards: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if any(len(sample) != 3 for sample in self.samples):
            raise ValueError("samples must be (play_time_s, field_x_yards, field_y_yards)")
        if any(not np.all(np.isfinite(np.asarray(sample, dtype=float))) for sample in self.samples):
            raise ValueError("field track samples must be finite")
        if any(right[0] <= left[0] for left, right in zip(self.samples, self.samples[1:])):
            raise ValueError("field track samples must be strictly ordered by play time")
        if self.uncertainty_yards and len(self.uncertainty_yards) != len(self.samples):
            raise ValueError("field track uncertainty must align with samples")
        if any(not np.isfinite(value) or value < 0 for value in self.uncertainty_yards):
            raise ValueError("field track uncertainty must be finite and non-negative")


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
    last_right_index = -1
    for time_s, x, y in left.samples:
        candidates = [(abs(float(right_times[index] - time_s)), index) for index in range(last_right_index + 1, len(right_times)) if abs(float(right_times[index] - time_s)) <= tolerance_s]
        if candidates:
            _, index = min(candidates, key=lambda item: (item[0], item[1]))
            last_right_index = index
            left_pairs.append((x, y))
            right_pairs.append(tuple(right_points[index]))
    return np.asarray(left_pairs, dtype=float), np.asarray(right_pairs, dtype=float)


def _paired_indices(left: FieldTrack, right: FieldTrack, tolerance_s: float) -> list[tuple[int, int]]:
    if not left.samples or not right.samples:
        return []
    right_times = np.asarray([sample[0] for sample in right.samples], dtype=float)
    last_right_index = -1
    pairs: list[tuple[int, int]] = []
    for left_index, sample in enumerate(left.samples):
        candidate_indices = range(last_right_index + 1, len(right_times))
        candidates = [(abs(float(right_times[index] - sample[0])), index) for index in candidate_indices if abs(float(right_times[index] - sample[0])) <= tolerance_s]
        if candidates:
            _, right_index = min(candidates, key=lambda item: (item[0], item[1]))
            last_right_index = right_index
            pairs.append((left_index, right_index))
    return pairs


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


def cross_shot_candidate_evidence(
    left: Sequence[FieldTrack],
    right: Sequence[FieldTrack],
    teams: Mapping[str, TeamEvidence],
    *,
    max_field_distance_yards: float = 6.0,
    min_overlap_samples: int = 5,
    sample_tolerance_s: float = 0.05,
    min_overlap_duration_s: float = 0.4,
    max_position_uncertainty_yards: float = 2.0,
) -> dict[tuple[str, str], dict[str, object]]:
    """Return auditable evidence for every possible cross-shot pair.

    Hard constraints are represented as a rejection reason.  The public score
    wrapper below omits those pairs before assignment, so a low score can never
    be mistaken for a weak ``same`` decision.
    """

    if max_field_distance_yards <= 0 or not np.isfinite(max_field_distance_yards) or min_overlap_samples < 2 or sample_tolerance_s <= 0 or not np.isfinite(sample_tolerance_s) or min_overlap_duration_s < 0 or not np.isfinite(min_overlap_duration_s) or max_position_uncertainty_yards < 0 or not np.isfinite(max_position_uncertainty_yards):
        raise ValueError("invalid cross-shot scoring constraints")
    left_ids = [track.tracklet_id for track in left]
    right_ids = [track.tracklet_id for track in right]
    if len(set(left_ids)) != len(left_ids) or len(set(right_ids)) != len(right_ids):
        raise ValueError("cross-shot candidate tracks must have unique tracklet ids per shot")
    evidence: dict[tuple[str, str], dict[str, object]] = {}
    for left_track in sorted(left, key=lambda track: track.tracklet_id):
        left_team = teams.get(left_track.tracklet_id)
        for right_track in sorted(right, key=lambda track: track.tracklet_id):
            key = (left_track.tracklet_id, right_track.tracklet_id)
            record: dict[str, object] = {
                "left_key": left_track.tracklet_id,
                "right_key": right_track.tracklet_id,
                "eligible": False,
                "score": None,
                "paired_sample_count": 0,
                "overlap_start_s": None,
                "overlap_end_s": None,
                "overlap_span_s": 0.0,
                "median_distance_yards": None,
                "p90_distance_yards": None,
                "shape_agreement": None,
                "combined_uncertainty_yards": None,
                "left_team": left_team.team if left_team is not None else "unknown",
                "right_team": None,
                "left_team_score": left_team.score if left_team is not None else 0.0,
                "right_team_score": 0.0,
                "rejection_reason": None,
            }
            right_team = teams.get(right_track.tracklet_id)
            record["right_team"] = right_team.team if right_team is not None else "unknown"
            record["right_team_score"] = right_team.score if right_team is not None else 0.0
            if left_team is None or not left_team.eligible:
                record["rejection_reason"] = "left_team_ineligible"
                evidence[key] = record
                continue
            if right_team is None or not right_team.eligible:
                record["rejection_reason"] = "right_team_ineligible"
                evidence[key] = record
                continue
            if left_team.team != right_team.team:
                record["rejection_reason"] = "team_mismatch"
                evidence[key] = record
                continue
            paired_indices = _paired_indices(left_track, right_track, sample_tolerance_s)
            record["paired_sample_count"] = len(paired_indices)
            left_points = np.asarray([left_track.samples[left_index][1:] for left_index, _ in paired_indices], dtype=float)
            right_points = np.asarray([right_track.samples[right_index][1:] for _, right_index in paired_indices], dtype=float)
            if len(left_points) < min_overlap_samples:
                record["rejection_reason"] = "insufficient_overlap_samples"
                evidence[key] = record
                continue
            paired_times = [float(left_track.samples[left_index][0]) for left_index, _ in paired_indices]
            if paired_times:
                record["overlap_start_s"] = min(paired_times)
                record["overlap_end_s"] = max(paired_times)
                record["overlap_span_s"] = max(paired_times) - min(paired_times)
            if paired_times and float(record["overlap_span_s"]) < min_overlap_duration_s:
                record["rejection_reason"] = "insufficient_overlap_duration"
                evidence[key] = record
                continue
            uncertainties = np.asarray([
                (left_track.uncertainty_yards[left_index] if left_track.uncertainty_yards else 0.0)
                + (right_track.uncertainty_yards[right_index] if right_track.uncertainty_yards else 0.0)
                for left_index, right_index in paired_indices
            ], dtype=float)
            record["combined_uncertainty_yards"] = float(np.median(uncertainties)) if uncertainties.size else 0.0
            if uncertainties.size and float(np.median(uncertainties)) > max_position_uncertainty_yards:
                record["rejection_reason"] = "position_uncertainty_exceeds_limit"
                evidence[key] = record
                continue
            distances = np.linalg.norm(left_points - right_points, axis=1)
            median_distance = float(np.median(distances))
            p90_distance = float(np.percentile(distances, 90))
            record["median_distance_yards"] = median_distance
            record["p90_distance_yards"] = p90_distance
            if p90_distance > max_field_distance_yards:
                record["rejection_reason"] = "field_distance_exceeds_limit"
                evidence[key] = record
                continue
            position = 1.0 - (median_distance / max_field_distance_yards)
            shape = _shape_agreement(left_points, right_points)
            record["shape_agreement"] = shape
            team_confidence = float(left_team.score * right_team.score)
            uncertainty_penalty = float(np.exp(-np.median(uncertainties) / max(1e-6, max_position_uncertainty_yards))) if uncertainties.size and max_position_uncertainty_yards > 0 else 1.0
            score = (0.55 * position + 0.30 * shape + 0.15 * team_confidence) * uncertainty_penalty
            record["score"] = max(0.0, min(1.0, score))
            record["eligible"] = True
            evidence[key] = record
    return evidence


def cross_shot_candidate_scores(
    left: Sequence[FieldTrack],
    right: Sequence[FieldTrack],
    teams: Mapping[str, TeamEvidence],
    *,
    max_field_distance_yards: float = 6.0,
    min_overlap_samples: int = 5,
    sample_tolerance_s: float = 0.05,
    min_overlap_duration_s: float = 0.4,
    max_position_uncertainty_yards: float = 2.0,
) -> dict[tuple[str, str], float]:
    """Score only eligible cross-shot pairs from view-invariant evidence."""

    evidence = cross_shot_candidate_evidence(
        left,
        right,
        teams,
        max_field_distance_yards=max_field_distance_yards,
        min_overlap_samples=min_overlap_samples,
        sample_tolerance_s=sample_tolerance_s,
        min_overlap_duration_s=min_overlap_duration_s,
        max_position_uncertainty_yards=max_position_uncertainty_yards,
    )
    return {key: float(record["score"]) for key, record in evidence.items() if record["eligible"] and record["score"] is not None}
