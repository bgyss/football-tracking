"""Build deterministic, source-addressed review items from identity evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class IdentityReviewError(ValueError):
    """Raised when identity evidence cannot be mapped to source observations."""


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_observation_index(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    by_tracklet: dict[str, list[dict[str, Any]]] = {}
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                tracklet_id = str(row.get("tracklet_id", "")).strip()
                if not tracklet_id:
                    continue
                try:
                    frame = int(row["frame_index"])
                    pts = int(row["pts"])
                    time_base = json.loads(row["time_base"])
                    box = json.loads(row["bbox_xyxy_px"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise IdentityReviewError(f"invalid observation row for {tracklet_id}") from error
                if frame < 0 or pts < 0 or not isinstance(time_base, list) or len(time_base) != 2 or not isinstance(box, list) or len(box) != 4:
                    raise IdentityReviewError(f"invalid source coordinates for observation {tracklet_id}@{frame}")
                try:
                    time_base_values = [int(time_base[0]), int(time_base[1])]
                    box_values = [float(item) for item in box]
                except (TypeError, ValueError) as error:
                    raise IdentityReviewError(f"invalid time_base or box for observation {tracklet_id}@{frame}") from error
                if any(value <= 0 for value in time_base_values) or not all(math.isfinite(value) for value in box_values) or box_values[2] <= box_values[0] or box_values[3] <= box_values[1]:
                    raise IdentityReviewError(f"invalid time_base or box for observation {tracklet_id}@{frame}")
                try:
                    xy_yards = json.loads(row["field_xy_yards"]) if row.get("field_xy_yards") else None
                except (TypeError, ValueError, json.JSONDecodeError):
                    xy_yards = None
                by_tracklet.setdefault(tracklet_id, []).append({
                    "shot_id": str(row.get("shot_id", "")),
                    "source_frame": frame,
                    "source_pts": pts,
                    "time_base": time_base_values,
                    "bbox_xyxy_px": box_values,
                    "detection_score": _optional_float(row.get("detection_score")),
                    "team": str(row.get("team", "unknown")),
                    "team_score": _optional_float(row.get("team_score")),
                    "field_xy_yards": xy_yards,
                    "position_uncertainty_yards": _optional_float(row.get("position_uncertainty_yards")),
                    "calibration_id": row.get("calibration_id") or None,
                    "calibration_status": row.get("calibration_status") or None,
                    "play_id": row.get("play_id") or None,
                    "play_time_s": _optional_float(row.get("play_time_s")),
                    "time_map_id": row.get("time_map_id") or None,
                })
    except OSError as error:
        raise IdentityReviewError(f"unable to read observations: {error}") from error
    for tracklet_id, rows in by_tracklet.items():
        rows.sort(key=lambda row: (row["source_frame"], row["source_pts"]))
        if len({row["source_frame"] for row in rows}) != len(rows):
            raise IdentityReviewError(f"duplicate source frame in observations for {tracklet_id}")
    return by_tracklet


def _nearest(rows: Sequence[Mapping[str, Any]], target_s: float | None, *, max_delta_s: float = 0.1) -> Mapping[str, Any] | None:
    if not rows:
        return None
    if target_s is not None:
        timed = [row for row in rows if _optional_float(row.get("play_time_s")) is not None]
        if not timed:
            return None
        nearest = min(timed, key=lambda row: (abs(float(row["play_time_s"]) - target_s), row["source_frame"]))
        return nearest if abs(float(nearest["play_time_s"]) - target_s) <= max_delta_s else None
    return rows[len(rows) // 2]


def _candidate_id(play_id: str, left: str | None, right: str | None, kind: str) -> str:
    value = "\0".join((play_id, left or "", right or "", kind))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _priority(kind: str, link: Mapping[str, Any] | None, evidence: Mapping[str, Any] | None) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if kind == "ambiguous":
        value = 0.92
        reasons.append("assignment_ambiguity")
    elif kind == "accepted":
        value = 0.72
        reasons.append("accepted_link_needs_audit")
    elif kind == "unmatched":
        value = 0.52
        reasons.append("unmatched_tracklet")
    else:
        value = 0.28
        reasons.append("candidate_rejection")
    if link is not None:
        margin = _optional_float(link.get("ambiguity_margin"))
        score = _optional_float(link.get("score"))
        if margin is not None:
            value += 0.2 * (1.0 - min(1.0, margin / 0.5))
            reasons.append("low_assignment_margin") if margin < 0.1 else None
        if score is not None:
            value += 0.1 * (1.0 - abs(score - 0.65) / 0.65)
        if link.get("rejection_reason"):
            reasons.append(str(link["rejection_reason"]))
    if evidence is not None:
        review_rank_score = _optional_float(evidence.get("review_rank_score"))
        if review_rank_score is not None:
            value += 0.15 * review_rank_score
            reasons.append("multi_cue_review_rank")
        cue_evidence = evidence.get("identity_cues")
        if isinstance(cue_evidence, Mapping) and cue_evidence.get("reviewed_jersey_conflict"):
            value = max(value, 0.96)
            reasons.append("reviewed_jersey_conflict")
        missing = [key for key in ("left_team", "right_team", "paired_sample_count", "combined_uncertainty_yards") if evidence.get(key) in (None, "unknown", 0)]
        if missing:
            value += min(0.08, 0.02 * len(missing))
            reasons.extend(f"missing_{key}" for key in missing)
        overlap = _optional_float(evidence.get("overlap_span_s"))
        if overlap is not None and overlap < 1.0:
            value += 0.03
            reasons.append("short_overlap")
        rejection = evidence.get("rejection_reason")
        if rejection:
            reasons.append(str(rejection))
    return round(max(0.0, min(1.0, value)), 6), sorted(set(reasons))


def build_identity_review_queue(
    identity_links: Mapping[str, Any],
    observations: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_sha256: str | None = None,
    max_items: int = 100,
    samples_per_pair: int = 3,
) -> dict[str, Any]:
    if max_items < 1 or samples_per_pair < 1:
        raise IdentityReviewError("max_items and samples_per_pair must be positive")
    declared_hash = identity_links.get("source_sha256")
    if source_sha256 is not None and declared_hash not in (None, source_sha256):
        raise IdentityReviewError("identity evidence source hash does not match the requested source")
    play_id = str(identity_links.get("play_id") or identity_links.get("analysis", {}).get("play_id") or "unassigned-play")
    links_raw = identity_links.get("links", [])
    evidence_raw = identity_links.get("candidate_evidence", [])
    if not isinstance(links_raw, list) or not isinstance(evidence_raw, list):
        raise IdentityReviewError("identity evidence links and candidate_evidence must be lists")
    links: dict[tuple[str, str | None], Mapping[str, Any]] = {}
    for raw in links_raw:
        if not isinstance(raw, Mapping):
            continue
        left = str(raw.get("left_key", ""))
        right_value = raw.get("right_key")
        right = None if right_value is None else str(right_value)
        if left:
            links[(left, right)] = raw
    candidates: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in evidence_raw:
        if not isinstance(raw, Mapping):
            continue
        left, right = str(raw.get("left_key", "")), str(raw.get("right_key", ""))
        if left and right:
            candidates[(left, right)] = raw
    pair_keys = set(candidates) | {(left, right) for left, right in links if right is not None}
    items: list[dict[str, Any]] = []
    for left, right in sorted(pair_keys):
        link = links.get((left, right))
        evidence = candidates.get((left, right))
        decision = str(link.get("decision", "candidate")) if link else "candidate"
        kind = "accepted" if decision == "same" else "rejected" if decision == "different" else "ambiguous" if decision == "insufficient_evidence" and right is not None else "rejected" if evidence is not None and not evidence.get("eligible", False) else "ambiguous"
        score = _optional_float((link or {}).get("score"))
        if score is None and evidence is not None:
            score = _optional_float(evidence.get("review_rank_score")) or _optional_float(evidence.get("score"))
        left_rows = list(observations.get(left, ()))
        right_rows = list(observations.get(right, ())) if right is not None else []
        start = _optional_float(evidence.get("overlap_start_s")) if evidence is not None else None
        end = _optional_float(evidence.get("overlap_end_s")) if evidence is not None else None
        sample_times: list[float | None]
        if start is not None and end is not None and end >= start:
            count = min(samples_per_pair, max(1, int((end - start) / 0.2) + 1))
            if count == 1:
                sample_times = [(start + end) / 2.0]
            else:
                sample_times = [start + (end - start) * index / (count - 1) for index in range(count)]
        else:
            sample_times = [None]
        samples: list[dict[str, Any]] = []
        seen_frames: set[tuple[int | None, int | None]] = set()
        for target in sample_times:
            left_row = _nearest(left_rows, target)
            right_row = _nearest(right_rows, target)
            pair = (left_row.get("source_frame") if left_row else None, right_row.get("source_frame") if right_row else None)
            if pair in seen_frames:
                continue
            seen_frames.add(pair)
            samples.append({
                "requested_play_time_s": target,
                "left": dict(left_row) if left_row else None,
                "right": dict(right_row) if right_row else None,
                "left_missing_reason": None if left_row is not None else "no_tracklet_observations" if not left_rows else "no_observation_within_time_tolerance",
                "right_missing_reason": None if right_row is not None else "no_tracklet_observations" if not right_rows else "no_observation_within_time_tolerance",
            })
        priority, reasons = _priority(kind, link, evidence)
        item_id = _candidate_id(play_id, left, right, kind)
        items.append({
            "review_item_id": item_id,
            "play_id": play_id,
            "kind": kind,
            "decision": decision,
            "review_status": "unreviewed",
            "priority": priority,
            "priority_reasons": reasons,
            "left_tracklet_id": left,
            "right_tracklet_id": right,
            "score": score,
            "alternative_score": _optional_float((link or {}).get("alternative_score")),
            "ambiguity_margin": _optional_float((link or {}).get("ambiguity_margin")),
            "rejection_reason": (link or {}).get("rejection_reason") or (evidence or {}).get("rejection_reason"),
            "evidence": dict(evidence) if evidence is not None else None,
            "evidence_keys": list((link or {}).get("evidence_keys", [])),
            "samples": samples,
            "source_frames_available": any(sample["left"] is not None and sample["right"] is not None for sample in samples) if right is not None else bool(left_rows),
        })
    for (left, right), link in sorted(links.items()):
        if right is None:
            priority, reasons = _priority("unmatched", link, None)
            rows = list(observations.get(left, ()))
            middle = _nearest(rows, None)
            items.append({
                "review_item_id": _candidate_id(play_id, left, None, "unmatched"),
                "play_id": play_id,
                "kind": "unmatched",
                "decision": link.get("decision", "insufficient_evidence"),
                "review_status": "unreviewed",
                "priority": priority,
                "priority_reasons": reasons,
                "left_tracklet_id": left,
                "right_tracklet_id": None,
                "score": _optional_float(link.get("score")),
                "alternative_score": _optional_float(link.get("alternative_score")),
                "ambiguity_margin": _optional_float(link.get("ambiguity_margin")),
                "rejection_reason": link.get("rejection_reason"),
                "evidence": None,
                "evidence_keys": list(link.get("evidence_keys", [])),
                "samples": [{"requested_play_time_s": None, "left": dict(middle) if middle else None, "right": None}],
                "source_frames_available": bool(middle),
            })
    items.sort(key=lambda item: (-item["priority"], item["play_id"], item["left_tracklet_id"], item["right_tracklet_id"] or "", item["review_item_id"]))
    return {
        "schema_version": 1,
        "reviewed": False,
        "source_sha256": source_sha256 or declared_hash,
        "play_id": play_id,
        "status": "proposal_queue",
        "queue_policy": "heuristic_priority_only; never ground truth",
        "total_item_count": len(items),
        "emitted_item_count": min(len(items), max_items),
        "items": items[:max_items],
    }
