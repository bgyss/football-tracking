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


class EvaluationError(ValueError):
    """Raised when a reference cannot support identity scoring."""


@dataclass(frozen=True, slots=True)
class ReferenceObject:
    identifier: str
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ReferenceFrame:
    source_frame: int
    evaluator_frame: int
    labeled: bool
    ignore: bool
    objects: tuple[ReferenceObject, ...]


def load_reviewed_mot_reference(path: str | Path) -> dict[str, dict[int, ReferenceFrame]]:
    """Load the project JSON interchange form emitted from reviewed CVAT/MOT data."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"unable to read reference: {error}") from error
    if not isinstance(value, dict) or value.get("reviewed") is not True:
        raise EvaluationError("reference must be explicitly marked reviewed: true")
    sequences = value.get("sequences")
    if not isinstance(sequences, dict) or not sequences:
        raise EvaluationError("reviewed reference has no sequences")
    result: dict[str, dict[int, ReferenceFrame]] = {}
    for shot_id, sequence in sequences.items():
        frames = sequence.get("frames") if isinstance(sequence, dict) else None
        if not isinstance(frames, dict):
            raise EvaluationError(f"sequence {shot_id!r} has no frame map")
        parsed: dict[int, ReferenceFrame] = {}
        for raw_index, raw_frame in frames.items():
            if not isinstance(raw_frame, dict):
                raise EvaluationError(f"invalid frame {raw_index!r}")
            source_frame = int(raw_index)
            if source_frame < 0 or source_frame in parsed:
                raise EvaluationError(f"invalid or duplicate source frame {source_frame}")
            objects: list[ReferenceObject] = []
            for raw_object in raw_frame.get("objects", []):
                if not isinstance(raw_object, dict) or "id" not in raw_object or "bbox_xyxy" not in raw_object:
                    raise EvaluationError(f"invalid object in {shot_id}:{source_frame}")
                box = tuple(float(item) for item in raw_object["bbox_xyxy"])
                if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                    raise EvaluationError(f"invalid box in {shot_id}:{source_frame}")
                objects.append(ReferenceObject(str(raw_object["id"]), box))
            parsed[source_frame] = ReferenceFrame(
                source_frame=source_frame,
                evaluator_frame=source_frame + 1,
                labeled=bool(raw_frame.get("labeled", False)),
                ignore=bool(raw_frame.get("ignore", False)),
                objects=tuple(objects),
            )
        result[str(shot_id)] = parsed
    return result


def load_cross_shot_identity(path: str | Path) -> dict[str, dict[str, str]] | None:
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
        result[str(shot_id)] = {str(key): str(name) for key, name in mapping.items()}
    return result


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
        edges: list[tuple[float, str, str, int, int]] = []
        for object_index, reference_object in enumerate(frame.objects):
            for prediction_index, prediction in enumerate(candidates):
                score = _iou(np.asarray(prediction.bbox_xyxy_px, dtype=float), np.asarray(reference_object.bbox_xyxy, dtype=float))
                if score >= iou_threshold:
                    edges.append((score, reference_object.identifier, prediction.tracklet_id, object_index, prediction_index))
        assigned_objects: set[int] = set()
        assigned_predictions: set[int] = set()
        for _, _, _, object_index, prediction_index in sorted(edges, key=lambda edge: (-edge[0], edge[1], edge[2], edge[4])):
            if object_index in assigned_objects or prediction_index in assigned_predictions:
                continue
            assigned_objects.add(object_index)
            assigned_predictions.add(prediction_index)
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
    for shot_id, frames in reference.items():
        shot_map = cross_shot_identity.get(shot_id, {})
        votes = reference_attribution(by_shot.get(shot_id, []), frames, iou_threshold)
        for tracklet_id, counts in votes.items():
            total_votes = sum(counts.values())
            winner = min(sorted(counts), key=lambda key: (-counts[key], key))
            if total_votes and counts[winner] / total_votes < 0.75:
                mixed_tracklets.append(tracklet_id)
        for tracklet_id, reference_id in dominant_reference_ids(by_shot.get(shot_id, []), frames, iou_threshold).items():
            global_id = shot_map.get(reference_id)
            if global_id is not None:
                truth[tracklet_id] = global_id
    attributed = sorted(truth)
    # Define eligibility from reviewed truth, independent of detector output.
    # A shared player missed in one view must remain in the denominator.
    global_shots: dict[str, set[str]] = {}
    for shot_id, shot_map in cross_shot_identity.items():
        for global_id in shot_map.values():
            global_shots.setdefault(str(global_id), set()).add(str(shot_id))
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
    component_contamination = {
        player_id: {shot_id: sorted(tracklets) for shot_id, tracklets in shots.items() if len(tracklets) > 1}
        for player_id, shots in by_identity.items()
        if any(len(tracklets) > 1 for tracklets in shots.values())
    }
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
        "merged_pairs": merged,
        "true_merges": true_merges,
        "false_merges": false_merges,
        "coverage": correct_globals / len(eligible_globals) if eligible_globals else 0.0,
        "precision": (true_merges / merged) if merged else None,
        "gate": {
            "false_merges_zero": false_merges == 0,
            "components_clean": not component_contamination and accepted_unattributed_pairs == 0,
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
        "component_contamination": component_contamination,
        "accepted_unattributed_pairs": accepted_unattributed_pairs,
        "false_merge_examples": false_merge_examples,
        "missed_pair_examples": missed_examples,
    }


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
        edges: list[tuple[float, int, int]] = []
        for gt_index, groundtruth in enumerate(frame.objects):
            for pred_index, prediction in enumerate(candidates):
                score = _iou(groundtruth.bbox_xyxy, prediction.bbox_xyxy_px)
                if score >= iou_threshold:
                    edges.append((score, gt_index, pred_index))
        assigned_gt: set[int] = set()
        assigned_pred: set[int] = set()
        assigned_ids: dict[int, str] = {}
        for _, gt_index, pred_index in sorted(edges, reverse=True):
            if gt_index not in assigned_gt and pred_index not in assigned_pred:
                assigned_gt.add(gt_index)
                assigned_pred.add(pred_index)
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
    data = _trackeval_data(rows, frames)
    return _evaluate_trackeval_data(trackeval, data, iou_threshold)


def _evaluate_trackeval_data(trackeval: Any, data: Mapping[str, Any], iou_threshold: float) -> dict[str, Any]:
    hota = trackeval.metrics.HOTA().eval_sequence(data)
    identity = trackeval.metrics.Identity({"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}).eval_sequence(data)
    clear = trackeval.metrics.CLEAR({"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}).eval_sequence(data)
    return {
        "status": "evaluated",
        "package": "trackeval==1.1.0",
        "HOTA": float(np.mean(hota["HOTA"])),
        "DetA": float(np.mean(hota["DetA"])),
        "AssA": float(np.mean(hota["AssA"])),
        "IDF1": float(identity["IDF1"]),
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
