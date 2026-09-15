"""Reviewed MOT-style reference import and deterministic tracking metrics.

This module deliberately refuses model-generated or unlabeled references.  Its
frame mapping is reversible: source frame ``n`` becomes MOT/TrackEval frame
``n + 1`` while exports retain the source frame index and PTS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .tracking import TrackObservation, _iou
from .schema import Observation
from .field import validate_field_point


class EvaluationError(ValueError):
    """Raised when a reference cannot support identity scoring."""


@dataclass(frozen=True, slots=True)
class ReferenceObject:
    identifier: str
    bbox_xyxy: tuple[float, float, float, float]
    ground_contact_xy_yards: tuple[float, float] | None = None
    team: str | None = None

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("reference object identifier must be non-empty")
        if len(self.bbox_xyxy) != 4 or not np.all(np.isfinite(np.asarray(self.bbox_xyxy, dtype=float))) or self.bbox_xyxy[2] <= self.bbox_xyxy[0] or self.bbox_xyxy[3] <= self.bbox_xyxy[1]:
            raise ValueError("reference object bbox must be finite and have positive dimensions")
        if self.ground_contact_xy_yards is not None:
            try:
                validate_field_point(self.ground_contact_xy_yards)
            except (TypeError, ValueError) as error:
                raise ValueError("reference ground contact must lie on the canonical field") from error
        if self.team is not None and not self.team.strip():
            raise ValueError("reference team label must be non-empty")


@dataclass(frozen=True, slots=True)
class ReferenceFrame:
    source_frame: int
    evaluator_frame: int
    labeled: bool
    ignore: bool
    objects: tuple[ReferenceObject, ...]
    pts: int | None = None

    def __post_init__(self) -> None:
        if self.source_frame < 0 or self.evaluator_frame != self.source_frame + 1:
            raise ValueError("reference frame numbering is inconsistent")
        if self.pts is not None and self.pts < 0:
            raise ValueError("reference frame pts must be non-negative")


def restrict_reference_to_window(
    reference: Mapping[str, Mapping[int, ReferenceFrame]],
    start_frame: int,
    end_frame: int,
) -> dict[str, dict[int, ReferenceFrame]]:
    """Return only reviewed frames covered by a source-frame analysis window."""

    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("invalid reference frame window")
    return {shot_id: {frame_index: frame for frame_index, frame in sorted(frames.items()) if start_frame <= frame_index < end_frame} for shot_id, frames in reference.items()}


def load_reviewed_mot_reference(path: str | Path, *, source_sha256: str | None = None) -> dict[str, dict[int, ReferenceFrame]]:
    """Load the project JSON interchange form emitted from reviewed CVAT/MOT data."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"unable to read reference: {error}") from error
    if not isinstance(value, dict) or value.get("reviewed") is not True:
        raise EvaluationError("reference must be explicitly marked reviewed: true")
    declared_source = value.get("source_sha256")
    if source_sha256 is not None and declared_source is not None and str(declared_source) != source_sha256:
        raise EvaluationError("reference source sha256 does not match the input video")
    if source_sha256 is not None and not declared_source:
        raise EvaluationError("reviewed reference must carry source_sha256 when used for a run")
    sequences = value.get("sequences")
    if not isinstance(sequences, dict) or not sequences:
        raise EvaluationError("reviewed reference has no sequences")
    result: dict[str, dict[int, ReferenceFrame]] = {}
    for shot_id, sequence in sequences.items():
        frames = sequence.get("frames") if isinstance(sequence, dict) else None
        if not isinstance(frames, dict):
            raise EvaluationError(f"sequence {shot_id!r} has no frame map")
        parsed: dict[int, ReferenceFrame] = {}
        previous_pts: int | None = None
        try:
            ordered_frames = sorted(frames.items(), key=lambda item: int(item[0]))
        except (TypeError, ValueError) as error:
            raise EvaluationError(f"sequence {shot_id!r} has an invalid source frame key") from error
        for raw_index, raw_frame in ordered_frames:
            if not isinstance(raw_frame, dict):
                raise EvaluationError(f"invalid frame {raw_index!r}")
            try:
                source_frame = int(raw_index)
            except (TypeError, ValueError) as error:
                raise EvaluationError(f"invalid source frame {raw_index!r}") from error
            if source_frame < 0 or source_frame in parsed:
                raise EvaluationError(f"invalid or duplicate source frame {source_frame}")
            objects: list[ReferenceObject] = []
            raw_objects = raw_frame.get("objects", [])
            if not isinstance(raw_objects, list):
                raise EvaluationError(f"objects for {shot_id}:{source_frame} must be a list")
            object_ids: set[str] = set()
            for raw_object in raw_objects:
                if not isinstance(raw_object, dict) or "id" not in raw_object or "bbox_xyxy" not in raw_object:
                    raise EvaluationError(f"invalid object in {shot_id}:{source_frame}")
                identifier = str(raw_object["id"]).strip()
                if not identifier or identifier in object_ids:
                    raise EvaluationError(f"duplicate or empty object id in {shot_id}:{source_frame}")
                object_ids.add(identifier)
                try:
                    box = tuple(float(item) for item in raw_object["bbox_xyxy"])
                except (TypeError, ValueError) as error:
                    raise EvaluationError(f"invalid box in {shot_id}:{source_frame}") from error
                if len(box) != 4 or not np.all(np.isfinite(np.asarray(box, dtype=float))) or box[2] <= box[0] or box[3] <= box[1]:
                    raise EvaluationError(f"invalid box in {shot_id}:{source_frame}")
                contact = raw_object.get("ground_contact_xy_yards")
                if contact is not None:
                    try:
                        contact = tuple(float(item) for item in contact)
                    except (TypeError, ValueError) as error:
                        raise EvaluationError(f"invalid ground contact in {shot_id}:{source_frame}") from error
                    if len(contact) != 2 or not np.all(np.isfinite(np.asarray(contact, dtype=float))):
                        raise EvaluationError(f"invalid ground contact in {shot_id}:{source_frame}")
                team = None if raw_object.get("team") is None else str(raw_object["team"])
                try:
                    objects.append(ReferenceObject(identifier, box, contact, team))
                except ValueError as error:
                    raise EvaluationError(f"invalid object in {shot_id}:{source_frame}") from error
            raw_pts = raw_frame.get("pts")
            try:
                pts = None if raw_pts is None else int(raw_pts)
            except (TypeError, ValueError) as error:
                raise EvaluationError(f"invalid pts in {shot_id}:{source_frame}") from error
            if pts is not None and pts < 0:
                raise EvaluationError(f"invalid pts in {shot_id}:{source_frame}")
            if pts is not None and previous_pts is not None and pts <= previous_pts:
                raise EvaluationError(f"non-monotonic pts in {shot_id}:{source_frame}")
            if pts is not None:
                previous_pts = pts
            parsed[source_frame] = ReferenceFrame(
                source_frame=source_frame,
                evaluator_frame=source_frame + 1,
                labeled=bool(raw_frame.get("labeled", False)),
                ignore=bool(raw_frame.get("ignore", False)),
                objects=tuple(objects),
                pts=pts,
            )
        result[str(shot_id)] = parsed
    return result


def load_cross_shot_identity(path: str | Path, *, source_sha256: str | None = None) -> dict[str, dict[str, str]] | None:
    """Load the optional reviewed map from sequence-local object ids to global player ids.

    Returns None when the reviewed reference simply does not carry the map, so a
    caller can report ``not_evaluated`` rather than inventing a cross-shot score.
    """

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"unable to read reference: {error}") from error
    if not isinstance(value, dict) or value.get("reviewed") is not True:
        raise EvaluationError("reference must be explicitly marked reviewed: true")
    declared_source = value.get("source_sha256")
    if source_sha256 is not None and (not declared_source or str(declared_source) != source_sha256):
        raise EvaluationError("reference source sha256 does not match the input video")
    raw = value.get("cross_shot_identity")
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise EvaluationError("cross_shot_identity must be a non-empty object when present")
    result: dict[str, dict[str, str]] = {}
    for shot_id, mapping in raw.items():
        if not isinstance(mapping, dict) or not mapping:
            raise EvaluationError(f"cross_shot_identity[{shot_id!r}] must be a non-empty object")
        # A global identity may intentionally have multiple non-overlapping
        # reviewed fragments in one shot. Component-level overlap checks are
        # reported during evaluation rather than rejecting the reference here.
        result[str(shot_id)] = {str(key): str(name) for key, name in mapping.items() if str(key).strip() and str(name).strip()}
        if len(result[str(shot_id)]) != len(mapping):
            raise EvaluationError(f"cross_shot_identity[{shot_id!r}] contains empty ids")
    return result


def _injective_matches(
    objects: Sequence[ReferenceObject],
    predictions: Sequence[TrackObservation],
    iou_threshold: float,
) -> tuple[tuple[int, int, float], ...]:
    """Associate reference objects and predictions at most once per frame.

    A dummy column is added for every reference object. Consequently a
    prediction is only used when it clears the IoU threshold; unmatched
    objects do not force a low-quality real association. Sorting the inputs
    makes equal-score outcomes independent of input ordering while returned
    indices still refer to the original lists.
    """

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    ordered_objects = sorted(enumerate(objects), key=lambda item: (item[1].identifier, item[0]))
    ordered_predictions = sorted(enumerate(predictions), key=lambda item: (item[1].tracklet_id, item[1].pts, item[1].frame_index, item[0]))
    if not ordered_objects or not ordered_predictions:
        return ()
    matrix = np.full((len(ordered_objects), len(ordered_predictions)), -1.0, dtype=float)
    for object_row, (_, reference_object) in enumerate(ordered_objects):
        reference_box = np.asarray(reference_object.bbox_xyxy, dtype=float)
        for prediction_column, (_, prediction) in enumerate(ordered_predictions):
            score = _iou(reference_box, np.asarray(prediction.bbox_xyxy_px, dtype=float))
            if np.isfinite(score) and score >= iou_threshold:
                matrix[object_row, prediction_column] = score
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise RuntimeError("scipy is required for deterministic reference attribution") from error
    assignment_values = np.concatenate([matrix, np.zeros((len(ordered_objects), len(ordered_objects)), dtype=float)], axis=1)
    rows, columns = linear_sum_assignment(-assignment_values)
    matches: list[tuple[int, int, float]] = []
    for object_row, prediction_column in zip(rows, columns):
        if prediction_column >= len(ordered_predictions) or matrix[object_row, prediction_column] < 0:
            continue
        object_index = ordered_objects[object_row][0]
        prediction_index = ordered_predictions[prediction_column][0]
        matches.append((object_index, prediction_index, float(matrix[object_row, prediction_column])))
    return tuple(sorted(matches, key=lambda match: (match[0], match[1])))


def reference_attribution(
    rows: Sequence[TrackObservation],
    frames: Mapping[int, ReferenceFrame],
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, int]]:
    """Return injective per-frame reference votes for every tracklet."""

    votes: dict[str, dict[str, int]] = {}
    rows_by_frame: dict[int, list[TrackObservation]] = {}
    for row in rows:
        rows_by_frame.setdefault(row.frame_index, []).append(row)
    for frame_index, candidates in rows_by_frame.items():
        frame = frames.get(frame_index)
        if frame is None or not frame.labeled or frame.ignore:
            continue
        for object_index, prediction_index, _ in _injective_matches(frame.objects, candidates, iou_threshold):
            tracklet_id = candidates[prediction_index].tracklet_id
            reference_id = frame.objects[object_index].identifier
            votes.setdefault(tracklet_id, {}).setdefault(reference_id, 0)
            votes[tracklet_id][reference_id] += 1
    return votes


def dominant_reference_ids(
    rows: Sequence[TrackObservation],
    frames: Mapping[int, ReferenceFrame],
    iou_threshold: float = 0.5,
    *,
    min_purity: float = 0.75,
) -> dict[str, str]:
    """Attribute pure tracklets to their dominant reviewed reference object."""

    if not 0.0 < min_purity <= 1.0:
        raise ValueError("min_purity must be in (0, 1]")
    votes = reference_attribution(rows, frames, iou_threshold)
    result: dict[str, str] = {}
    for tracklet_id, counts in votes.items():
        total = sum(counts.values())
        winner = min(sorted(counts), key=lambda key: (-counts[key], key))
        if total and counts[winner] / total >= min_purity:
            result[tracklet_id] = winner
    return result


def evaluate_cross_shot_identity(
    identity_map: Mapping[str, str],
    predictions: Sequence[TrackObservation],
    reference: Mapping[str, Mapping[int, ReferenceFrame]],
    cross_shot_identity: Mapping[str, Mapping[str, str]] | None,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score cross-shot merges as precision and coverage, never as a single number."""

    if not cross_shot_identity:
        return {"status": "not_evaluated", "reason": "reviewed reference carries no cross_shot_identity map"}
    by_shot: dict[str, list[TrackObservation]] = {}
    for row in predictions:
        by_shot.setdefault(row.tracklet_id.split(":", 1)[0], []).append(row)
    truth: dict[str, str] = {}
    mixed_tracklets: list[str] = []
    attribution_diagnostics: dict[str, dict[str, object]] = {}
    for shot_id, frames in reference.items():
        shot_map = cross_shot_identity.get(shot_id, {})
        votes = reference_attribution(by_shot.get(shot_id, []), frames, iou_threshold)
        for tracklet_id, counts in votes.items():
            total_votes = sum(counts.values())
            winner = min(sorted(counts), key=lambda key: (-counts[key], key))
            purity = counts[winner] / total_votes if total_votes else 0.0
            attribution_diagnostics[tracklet_id] = {"reference_id": winner, "votes": dict(sorted(counts.items())), "purity": purity}
            if total_votes and purity < 0.75:
                mixed_tracklets.append(tracklet_id)
        for tracklet_id, reference_id in dominant_reference_ids(by_shot.get(shot_id, []), frames, iou_threshold).items():
            global_id = shot_map.get(reference_id)
            if global_id is not None:
                truth[tracklet_id] = global_id
    attributed = sorted(truth)
    # Define eligibility from reviewed truth, independent of detector output.
    # A shared player missed in one view must remain in the denominator.
    visible_reference_ids: dict[str, set[str]] = {
        shot_id: {
            obj.identifier
            for frame in frames.values()
            if frame.labeled and not frame.ignore
            for obj in frame.objects
        }
        for shot_id, frames in reference.items()
    }
    global_shots: dict[str, set[str]] = {}
    for shot_id, shot_map in cross_shot_identity.items():
        if not visible_reference_ids.get(str(shot_id)):
            continue
        for reference_id, global_id in shot_map.items():
            # Ignore map entries whose reference object is outside the scored
            # frame window; bounded runs must not be penalized for unseen shots.
            normalized_global_id = str(global_id).strip()
            if normalized_global_id.lower() in {"", "unknown", "unresolved", "ambiguous"}:
                continue
            if str(reference_id) in visible_reference_ids.get(str(shot_id), set()):
                global_shots.setdefault(normalized_global_id, set()).add(str(shot_id))
    eligible_globals = sorted(global_id for global_id, shots in global_shots.items() if len(shots) >= 2)
    if not eligible_globals:
        return {"status": "not_evaluated", "reason": "reviewed cross_shot_identity map has no player present in at least two shots"}
    tracklets_by_global: dict[str, list[str]] = {global_id: [] for global_id in eligible_globals}
    for tracklet_id, global_id in truth.items():
        if global_id in tracklets_by_global:
            tracklets_by_global[global_id].append(tracklet_id)
    correct_globals = 0
    missing_globals: list[str] = []
    for global_id in eligible_globals:
        tracklets = tracklets_by_global[global_id]
        observed_shots = {tracklet.split(":", 1)[0] for tracklet in tracklets}
        predicted_ids = {identity_map.get(tracklet) for tracklet in tracklets if identity_map.get(tracklet) is not None}
        if len(observed_shots) >= 2 and len(predicted_ids) == 1:
            correct_globals += 1
        else:
            missing_globals.append(global_id)
    resolvable = 0
    merged = 0
    true_merges = 0
    false_merges = 0
    false_merge_examples: list[dict[str, str]] = []
    missed_examples: list[dict[str, str]] = []
    for index, left in enumerate(attributed):
        for right in attributed[index + 1 :]:
            if left.split(":", 1)[0] == right.split(":", 1)[0]:
                continue
            same_player = truth[left] == truth[right]
            same_id = identity_map.get(left) is not None and identity_map.get(left) == identity_map.get(right)
            if same_player:
                resolvable += 1
            if same_id:
                merged += 1
            if same_id and same_player:
                true_merges += 1
            elif same_id and not same_player:
                false_merges += 1
                if len(false_merge_examples) < 20:
                    false_merge_examples.append({"left": left, "right": right, "left_player": truth[left], "right_player": truth[right]})
            elif same_player and not same_id:
                if len(missed_examples) < 20:
                    missed_examples.append({"left": left, "right": right, "player": truth[left]})
    by_identity: dict[str, dict[str, set[str]]] = {}
    for tracklet_id, player_id in identity_map.items():
        shot_id = tracklet_id.split(":", 1)[0]
        by_identity.setdefault(player_id, {}).setdefault(shot_id, set()).add(tracklet_id)
    frame_sets: dict[str, set[int]] = {}
    for row in predictions:
        frame_sets.setdefault(row.tracklet_id, set()).add(row.frame_index)
    component_contamination: dict[str, dict[str, list[str]]] = {}
    for player_id, shots in by_identity.items():
        collisions: dict[str, list[str]] = {}
        for shot_id, tracklets in shots.items():
            overlapping = {tracklet for tracklet in tracklets if any(frame_sets.get(tracklet, set()) & frame_sets.get(other, set()) for other in tracklets if other != tracklet)}
            reviewed_globals = {truth[tracklet] for tracklet in tracklets if tracklet in truth}
            if overlapping or len(reviewed_globals) > 1:
                collisions[shot_id] = sorted(tracklets)
        if collisions:
            component_contamination[player_id] = collisions
    all_identity_tracklets = sorted(identity_map)
    accepted_unattributed_pairs = sum(
        (left not in truth or right not in truth)
        for index, left in enumerate(all_identity_tracklets)
        for right in all_identity_tracklets[index + 1 :]
        if left.split(":", 1)[0] != right.split(":", 1)[0]
        and identity_map.get(left) == identity_map.get(right)
    )
    return {
        "status": "evaluated",
        "iou_threshold": iou_threshold,
        "attributed_tracklets": len(truth),
        # Kept for compatibility with the earlier report; it now means the
        # number of reviewed shared players, rather than prediction-derived
        # tracklet pairs.
        "resolvable_pairs": len(eligible_globals),
        "resolvable_tracklet_pairs": resolvable,
        "merged_pairs": merged,
        "true_merges": true_merges,
        "false_merges": false_merges,
        "coverage": correct_globals / len(eligible_globals) if eligible_globals else 0.0,
        "precision": (true_merges / merged) if merged else None,
        "gate": {
            "false_merges_zero": false_merges == 0,
            "components_clean": not component_contamination and accepted_unattributed_pairs == 0 and not mixed_tracklets,
            "coverage_at_least_0_80": (correct_globals / len(eligible_globals) if eligible_globals else 0.0) >= 0.80,
        },
        "shared_player_coverage": {
            "correct": correct_globals,
            "eligible": len(eligible_globals),
            "value": correct_globals / len(eligible_globals) if eligible_globals else 0.0,
        },
        "eligible_shared_players": eligible_globals,
        "missing_shared_players": missing_globals,
        "unattributed_tracklets": sorted(set(identity_map) - set(truth)),
        "mixed_tracklets": sorted(set(mixed_tracklets)),
        "attribution": attribution_diagnostics,
        "component_contamination": component_contamination,
        "accepted_unattributed_pairs": accepted_unattributed_pairs,
        "component_errors": {"contamination": component_contamination, "accepted_unattributed_pairs": accepted_unattributed_pairs, "mixed_tracklets": sorted(set(mixed_tracklets))},
        "false_merge_examples": false_merge_examples,
        "missed_pair_examples": missed_examples,
    }


def evaluate_ground_contact_positions(
    observations: Sequence[Observation],
    reference: Mapping[str, Mapping[int, ReferenceFrame]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Measure calibrated bottom-center/contact positions against reviewed contacts."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    by_frame: dict[tuple[str, int], list[Observation]] = {}
    for observation in observations:
        by_frame.setdefault((observation.shot_id, observation.frame_index), []).append(observation)
    errors: list[float] = []
    evaluated = 0
    invalid = 0
    examples: list[dict[str, object]] = []
    for shot_id, frames in reference.items():
        for frame_index, frame in frames.items():
            contacts = [obj for obj in frame.objects if obj.ground_contact_xy_yards is not None]
            if not frame.labeled or frame.ignore or not contacts:
                continue
            predictions = by_frame.get((shot_id, frame_index), [])
            matched_contacts: set[int] = set()
            for contact_index, prediction_index, _ in _injective_matches(contacts, predictions, iou_threshold):
                matched_contacts.add(contact_index)
                predicted = predictions[prediction_index]
                contact = contacts[contact_index].ground_contact_xy_yards
                if predicted.field_xy_yards is None:
                    invalid += 1
                    continue
                error = float(np.linalg.norm(np.asarray(predicted.field_xy_yards) - np.asarray(contact)))
                if not np.isfinite(error):
                    invalid += 1
                    continue
                evaluated += 1
                errors.append(error)
                if len(examples) < 20:
                    examples.append({"shot_id": shot_id, "frame_index": frame_index, "reference_id": contacts[contact_index].identifier, "error_yards": error})
            invalid += len(contacts) - len(matched_contacts)
    if not evaluated:
        return {"status": "not_evaluated", "reason": "reference carries no matched ground-contact positions", "evaluated_contacts": 0, "invalid_or_missing": invalid}
    values = np.asarray(errors, dtype=float)
    median = float(np.median(values))
    p95 = float(np.percentile(values, 95))
    valid_fraction = evaluated / (evaluated + invalid) if evaluated + invalid else 0.0
    return {
        "status": "evaluated",
        "iou_threshold": iou_threshold,
        "evaluated_contacts": evaluated,
        "invalid_or_missing": invalid,
        "median_error_yards": median,
        "p95_error_yards": p95,
        "valid_fraction": valid_fraction,
        "gate": {"median_at_most_1_5_yards": median <= 1.5, "valid_fraction_at_least_0_90": valid_fraction >= 0.90},
        "examples": examples,
    }


def evaluate_team_assignment(
    observations: Sequence[Observation],
    reference: Mapping[str, Mapping[int, ReferenceFrame]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score reviewed team labels separately from cross-shot identity."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    by_frame: dict[tuple[str, int], list[Observation]] = {}
    for observation in observations:
        by_frame.setdefault((observation.shot_id, observation.frame_index), []).append(observation)
    assigned = correct = 0
    unknown_reference = 0
    examples: list[dict[str, object]] = []
    for shot_id, frames in reference.items():
        for frame_index, frame in frames.items():
            objects = [obj for obj in frame.objects if obj.team is not None and obj.team != "unknown"]
            if not frame.labeled or frame.ignore or not objects:
                continue
            predictions = by_frame.get((shot_id, frame_index), [])
            matched_objects: set[int] = set()
            for object_index, prediction_index, _ in _injective_matches(objects, predictions, iou_threshold):
                matched_objects.add(object_index)
                assigned += 1
                prediction = predictions[prediction_index]
                expected = objects[object_index].team
                if prediction.team == expected:
                    correct += 1
                if len(examples) < 20:
                    examples.append({"shot_id": shot_id, "frame_index": frame_index, "reference_id": objects[object_index].identifier, "expected_team": expected, "predicted_team": prediction.team})
            unknown_reference += len(objects) - len(matched_objects)
    if not assigned:
        return {"status": "not_evaluated", "reason": "reference carries no matched team labels", "assigned": 0, "unknown_or_missing": unknown_reference}
    accuracy = correct / assigned
    coverage = assigned / (assigned + unknown_reference) if assigned + unknown_reference else 0.0
    return {"status": "evaluated", "assigned": assigned, "correct": correct, "unknown_or_missing": unknown_reference, "accuracy": accuracy, "coverage": coverage, "gate": {"accuracy_at_least_0_98": accuracy >= 0.98, "coverage_at_least_0_95": coverage >= 0.95}, "examples": examples}


def _metrics_for_shot(rows: Sequence[TrackObservation], frames: Mapping[int, ReferenceFrame], iou_threshold: float) -> dict[str, Any]:
    predicted: dict[int, list[TrackObservation]] = {}
    for row in rows:
        predicted.setdefault(row.frame_index, []).append(row)
    total_gt = matches = false_positive = 0
    assignments: dict[str, list[str | None]] = {}
    for frame_index in sorted(frames):
        frame = frames[frame_index]
        if not frame.labeled or frame.ignore:
            continue
        candidates = predicted.get(frame_index, [])
        total_gt += len(frame.objects)
        assigned_ids: dict[int, str] = {}
        matches_for_frame = _injective_matches(frame.objects, candidates, iou_threshold)
        assigned_gt = {object_index for object_index, _, _ in matches_for_frame}
        assigned_pred = {prediction_index for _, prediction_index, _ in matches_for_frame}
        for gt_index, pred_index, _ in matches_for_frame:
            assigned_ids[gt_index] = candidates[pred_index].tracklet_id
        matches += len(assigned_gt)
        false_positive += len(candidates) - len(assigned_pred)
        for gt_index, groundtruth in enumerate(frame.objects):
            assignments.setdefault(groundtruth.identifier, []).append(assigned_ids.get(gt_index))
    switches = fragmentation = 0
    stable_edges = total_edges = 0
    for identifiers in assignments.values():
        previous: str | None = None
        observed_before_gap = False
        gap_after_observation = False
        for identifier in identifiers:
            if identifier is None:
                if observed_before_gap:
                    gap_after_observation = True
                continue
            if previous is not None:
                total_edges += 1
                stable_edges += identifier == previous
                switches += identifier != previous
            if gap_after_observation:
                fragmentation += 1
                gap_after_observation = False
            previous = identifier
            observed_before_gap = True
    coverage = matches / total_gt if total_gt else 0.0
    deta = matches / (total_gt + false_positive) if total_gt + false_positive else 0.0
    assa = stable_edges / total_edges if total_edges else (1.0 if matches else 0.0)
    return {
        "evaluated_frames": sum(frame.labeled and not frame.ignore for frame in frames.values()),
        "ground_truth_detections": total_gt,
        "matched_detections": matches,
        "false_positives": false_positive,
        "coverage": coverage,
        "DetA": deta,
        "AssA": assa,
        "HOTA": (deta * assa) ** 0.5,
        # This is detection F1, not identity F1. Standard IDF1 is supplied
        # only by TrackEval and is never substituted with this diagnostic.
        "detection_f1": (2 * matches) / (2 * matches + false_positive + (total_gt - matches)) if total_gt else 0.0,
        "id_switches": switches,
        "fragmentation": fragmentation,
    }


def _trackeval_data(rows: Sequence[TrackObservation], frames: Mapping[int, ReferenceFrame]) -> dict[str, Any]:
    """Translate the reviewed source-frame contract into TrackEval's sequence data."""

    relevant = [frame for _, frame in sorted(frames.items()) if frame.labeled and not frame.ignore]
    gt_names = sorted({object.identifier for frame in relevant for object in frame.objects})
    tracker_names = sorted({row.tracklet_id for row in rows if row.frame_index in {frame.source_frame for frame in relevant}})
    gt_index = {identifier: index for index, identifier in enumerate(gt_names)}
    tracker_index = {identifier: index for index, identifier in enumerate(tracker_names)}
    by_frame: dict[int, list[TrackObservation]] = {}
    for row in rows:
        by_frame.setdefault(row.frame_index, []).append(row)
    gt_ids: list[Any] = []
    tracker_ids: list[Any] = []
    similarities: list[Any] = []
    for frame in relevant:
        gt_objects = list(frame.objects)
        tracker_rows = by_frame.get(frame.source_frame, [])
        gt_ids.append(np.asarray([gt_index[object.identifier] for object in gt_objects], dtype=int))
        tracker_ids.append(np.asarray([tracker_index[row.tracklet_id] for row in tracker_rows], dtype=int))
        similarities.append(np.asarray([[_iou(object.bbox_xyxy, row.bbox_xyxy_px) for row in tracker_rows] for object in gt_objects], dtype=float).reshape(len(gt_objects), len(tracker_rows)))
    return {
        "num_timesteps": len(relevant),
        "num_gt_ids": len(gt_names),
        "num_tracker_ids": len(tracker_names),
        "num_gt_dets": sum(len(values) for values in gt_ids),
        "num_tracker_dets": sum(len(values) for values in tracker_ids),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarities,
    }


def _standard_trackeval_metrics(rows: Sequence[TrackObservation], frames: Mapping[int, ReferenceFrame], iou_threshold: float) -> dict[str, Any]:
    try:
        import trackeval  # type: ignore[import-not-found]
    except ImportError:
        return {"status": "unavailable", "reason": "install the evaluation optional dependency"}
    try:
        data = _trackeval_data(rows, frames)
        return _evaluate_trackeval_data(trackeval, data, iou_threshold)
    except Exception as error:
        return {"status": "unavailable", "reason": f"TrackEval evaluation failed: {type(error).__name__}: {error}"}


def _evaluate_trackeval_data(trackeval: Any, data: Mapping[str, Any], iou_threshold: float) -> dict[str, Any]:
    hota = trackeval.metrics.HOTA().eval_sequence(data)
    identity = trackeval.metrics.Identity({"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}).eval_sequence(data)
    clear = trackeval.metrics.CLEAR({"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}).eval_sequence(data)
    values = {
        "HOTA": float(np.mean(hota["HOTA"])),
        "DetA": float(np.mean(hota["DetA"])),
        "AssA": float(np.mean(hota["AssA"])),
        "IDF1": float(identity["IDF1"]),
    }
    if not all(np.isfinite(value) for value in values.values()):
        return {"status": "unavailable", "reason": "TrackEval returned non-finite standard metrics"}
    return {
        "status": "evaluated",
        "package": "trackeval==1.1.0",
        **values,
        "id_switches": int(clear["IDSW"]),
        "fragmentation": int(clear["Frag"]),
    }


def _combined_trackeval_data(
    rows_by_shot: Mapping[str, Sequence[TrackObservation]],
    reference: Mapping[str, Mapping[int, ReferenceFrame]],
) -> dict[str, Any]:
    """Concatenate shot sequences with disjoint IDs for one standard score."""

    datasets = [_trackeval_data(rows_by_shot.get(shot_id, ()), frames) for shot_id, frames in sorted(reference.items())]
    gt_offset = tracker_offset = 0
    gt_ids: list[Any] = []
    tracker_ids: list[Any] = []
    similarities: list[Any] = []
    for data in datasets:
        for values in data["gt_ids"]:
            gt_ids.append(values + gt_offset)
        for values in data["tracker_ids"]:
            tracker_ids.append(values + tracker_offset)
        similarities.extend(data["similarity_scores"])
        gt_offset += int(data["num_gt_ids"])
        tracker_offset += int(data["num_tracker_ids"])
    return {
        "num_timesteps": len(gt_ids),
        "num_gt_ids": gt_offset,
        "num_tracker_ids": tracker_offset,
        "num_gt_dets": sum(len(values) for values in gt_ids),
        "num_tracker_dets": sum(len(values) for values in tracker_ids),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarities,
    }


def evaluate_tracking(
    predictions: Sequence[TrackObservation],
    reference: Mapping[str, Mapping[int, ReferenceFrame]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score raw shot-local tracklets; unlabeled/ignored frames are excluded."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    by_shot: dict[str, list[TrackObservation]] = {}
    for row in predictions:
        by_shot.setdefault(row.tracklet_id.split(":", 1)[0], []).append(row)
    per_shot = {shot_id: _metrics_for_shot(by_shot.get(shot_id, []), frames, iou_threshold) for shot_id, frames in reference.items()}
    for shot_id, metrics in per_shot.items():
        metrics["trackeval"] = _standard_trackeval_metrics(by_shot.get(shot_id, []), reference[shot_id], iou_threshold)
    keys = ("evaluated_frames", "ground_truth_detections", "matched_detections", "false_positives", "id_switches", "fragmentation")
    aggregate: dict[str, Any] = {key: sum(int(report[key]) for report in per_shot.values()) for key in keys}
    gt = aggregate["ground_truth_detections"]
    matched = aggregate["matched_detections"]
    fp = aggregate["false_positives"]
    aggregate["coverage"] = matched / gt if gt else 0.0
    aggregate["DetA"] = matched / (gt + fp) if gt + fp else 0.0
    weighted_assa = sum(report["AssA"] * report["matched_detections"] for report in per_shot.values())
    aggregate["AssA"] = weighted_assa / matched if matched else 0.0
    aggregate["HOTA"] = (aggregate["DetA"] * aggregate["AssA"]) ** 0.5
    trackeval_reports = {shot_id: report["trackeval"] for shot_id, report in per_shot.items()}
    trackeval_available = bool(trackeval_reports) and all(report.get("status") == "evaluated" for report in trackeval_reports.values())
    standard_aggregate: dict[str, Any] | None = None
    if trackeval_available:
        try:
            import trackeval  # type: ignore[import-not-found]

            standard_aggregate = _evaluate_trackeval_data(trackeval, _combined_trackeval_data(by_shot, reference), iou_threshold)
        except Exception as error:
            trackeval_available = False
            standard_aggregate = {"status": "unavailable", "reason": f"TrackEval aggregation failed: {type(error).__name__}: {error}"}
    standard_metrics = {
        "status": "evaluated" if trackeval_available else "unavailable",
        "evaluator": "trackeval==1.1.0" if trackeval_available else None,
        "reason": None if trackeval_available else (standard_aggregate or {}).get("reason", "install the evaluation optional dependency"),
        "aggregate": standard_aggregate,
        "per_shot": trackeval_reports,
        "aggregation": "TrackEval sequence concatenation with disjoint IDs" if trackeval_available else "unavailable",
    }
    return {
        "schema_version": 2,
        "status": "evaluated",
        "evaluator": {"name": "trackeval-compatible", "frame_numbering": "source_frame + 1", "iou_threshold": iou_threshold},
        "per_shot": per_shot,
        "aggregate": aggregate,
        "diagnostics": {"aggregate": aggregate, "per_shot": {shot_id: {key: value for key, value in report.items() if key != "trackeval"} for shot_id, report in per_shot.items()}},
        "standard_metrics": standard_metrics,
    }


def evaluate_promotion_gates(
    tracking_report: Mapping[str, Any],
    cross_shot_report: Mapping[str, Any] | None,
    calibration_report: Mapping[str, Any],
    *,
    min_idf1: float = 0.90,
    max_id_switches_per_shot: int = 1,
    min_cross_shot_coverage: float = 0.80,
    ground_contact_report: Mapping[str, Any] | None = None,
    proxy_detector: bool = False,
    team_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an evidence-gated promotion decision without filling missing data."""

    if not 0.0 <= min_idf1 <= 1.0 or max_id_switches_per_shot < 0 or not 0.0 <= min_cross_shot_coverage <= 1.0:
        raise ValueError("invalid promotion gate thresholds")
    standard_value = tracking_report.get("standard_metrics", {})
    standard = standard_value if isinstance(standard_value, Mapping) else {}
    calibration_status = calibration_report.get("status")
    cross = cross_shot_report or {"status": "not_evaluated", "reason": "cross-shot report was not supplied"}
    reasons: list[str] = []
    missing_evidence = False
    if proxy_detector:
        reasons.append("proxy detector results cannot satisfy the promotion gate")
        missing_evidence = True
    if standard.get("status") != "evaluated":
        reasons.append("standard tracking metrics are unavailable")
        missing_evidence = True
    else:
        aggregate = standard.get("aggregate") or {}
        if aggregate.get("IDF1") is None or float(aggregate.get("IDF1", 0.0)) < min_idf1:
            reasons.append("IDF1 is below the promotion gate")
        per_shot = standard.get("per_shot", {})
        if any(report.get("IDF1") is None or float(report.get("IDF1", 0.0)) < min_idf1 for report in per_shot.values()):
            reasons.append("a shot is below the per-shot IDF1 gate")
        if any(int(report.get("id_switches", 0)) > max_id_switches_per_shot for report in per_shot.values()):
            reasons.append("a shot exceeds the identity-switch gate")
    if calibration_status in {None, "not_provided", "unvalidated", "partial"}:
        reasons.append(f"calibration status is {calibration_status or 'missing'}")
        missing_evidence = True
    elif calibration_status != "valid":
        reasons.append(f"calibration status is {calibration_status}")
    else:
        calibration_gate = calibration_report.get("gate")
        if not isinstance(calibration_gate, Mapping):
            reasons.append("calibration gate evidence is missing")
            missing_evidence = True
        elif not bool(calibration_gate.get("median_at_most_1_yard", False)) or not bool(calibration_gate.get("p95_at_most_2_yards", False)):
            reasons.append("calibration error is below the promotion gate")
    if cross.get("status") != "evaluated":
        reasons.append("cross-shot identity is not evaluated")
        missing_evidence = True
    else:
        cross_gate = cross.get("gate", {})
        if not isinstance(cross_gate, Mapping):
            cross_gate = {}
        if "false_merges_zero" not in cross_gate:
            reasons.append("cross-shot false-merge gate evidence is missing")
            missing_evidence = True
        elif int(cross.get("false_merges", 0)) != 0 or cross_gate.get("false_merges_zero") is not True:
            reasons.append("cross-shot identity has false merges")
        if "components_clean" not in cross_gate:
            reasons.append("identity component gate evidence is missing")
            missing_evidence = True
        elif cross_gate.get("components_clean") is not True:
            reasons.append("identity components are contaminated or unattributed")
        shared_coverage = cross.get("shared_player_coverage")
        coverage = float(cross.get("coverage", 0.0))
        if isinstance(shared_coverage, Mapping):
            try:
                eligible = int(shared_coverage.get("eligible", 0))
                correct = int(shared_coverage.get("correct", 0))
            except (TypeError, ValueError):
                eligible = correct = 0
            if eligible <= 0 or correct < 0 or correct > eligible:
                reasons.append("cross-shot shared-player denominator is invalid")
                missing_evidence = True
            else:
                coverage = correct / eligible
        else:
            reasons.append("cross-shot shared-player denominator is missing")
            missing_evidence = True
        if coverage < min_cross_shot_coverage:
            reasons.append("cross-shot coverage is below the promotion gate")
    if ground_contact_report is not None:
        if ground_contact_report.get("status") != "evaluated":
            reasons.append("ground-contact position error is not evaluated")
            missing_evidence = True
        else:
            contact_gate = ground_contact_report.get("gate")
            if not isinstance(contact_gate, Mapping):
                reasons.append("ground-contact gate evidence is missing")
                missing_evidence = True
            elif not bool(contact_gate.get("median_at_most_1_5_yards", False)) or not bool(contact_gate.get("valid_fraction_at_least_0_90", False)):
                reasons.append("ground-contact position error is below the promotion gate")
    if team_report is not None:
        if team_report.get("status") != "evaluated":
            reasons.append("team assignment is not evaluated")
            missing_evidence = True
        else:
            team_gate = team_report.get("gate")
            if not isinstance(team_gate, Mapping):
                reasons.append("team assignment gate evidence is missing")
                missing_evidence = True
            elif not bool(team_gate.get("accuracy_at_least_0_98", False)) or not bool(team_gate.get("coverage_at_least_0_95", False)):
                reasons.append("team assignment is below the promotion gate")
    status = "passed" if not reasons else "not_evaluated" if missing_evidence else "failed"
    return {"status": status, "thresholds": {"min_idf1": min_idf1, "max_id_switches_per_shot": max_id_switches_per_shot, "min_cross_shot_coverage": min_cross_shot_coverage}, "reasons": reasons}
