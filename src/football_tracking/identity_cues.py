"""Optional, source-hashed jersey and appearance cues for review ranking."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class IdentityCueError(ValueError):
    """Raised when optional identity cues have invalid provenance or values."""


@dataclass(frozen=True, slots=True)
class TrackletCues:
    tracklet_id: str
    jersey_reads: tuple[Mapping[str, Any], ...] = ()
    appearance_embeddings: tuple[Mapping[str, Any], ...] = ()


def load_tracklet_cues(path: str | Path, *, source_sha256: str, analysis_hash: str | None = None) -> dict[str, TrackletCues]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IdentityCueError(f"unable to read identity cue file: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise IdentityCueError("identity cue file must use schema_version 1")
    if str(value.get("source_sha256", "")) != source_sha256:
        raise IdentityCueError("identity cue source sha256 does not match input video")
    declared_analysis_hash = str(value.get("analysis_hash", ""))
    if len(declared_analysis_hash) != 16 or any(character not in "0123456789abcdef" for character in declared_analysis_hash.lower()):
        raise IdentityCueError("identity cue file must bind to a 16-character base analysis hash")
    if analysis_hash is not None and declared_analysis_hash != analysis_hash:
        raise IdentityCueError("identity cue analysis hash does not match current detector/tracker run")
    raw_tracklets = value.get("tracklets")
    if not isinstance(raw_tracklets, dict):
        raise IdentityCueError("identity cue tracklets must be an object")
    result: dict[str, TrackletCues] = {}
    seen_observations: set[tuple[str, str, int]] = set()
    for raw_id, raw in raw_tracklets.items():
        tracklet_id = str(raw_id).strip()
        if not tracklet_id or not isinstance(raw, dict):
            raise IdentityCueError("identity cue entries need a tracklet id and object")
        jersey_reads: list[Mapping[str, Any]] = []
        for reading in raw.get("jersey_reads", []):
            if not isinstance(reading, dict):
                raise IdentityCueError(f"jersey read for {tracklet_id} must be an object")
            try:
                frame, pts = int(reading["source_frame"]), int(reading["source_pts"])
                bbox = tuple(float(item) for item in reading["bbox_xyxy_px"])
                status = str(reading.get("review_status", "unreviewed"))
                candidates = reading.get("candidates", [])
            except (KeyError, TypeError, ValueError) as error:
                raise IdentityCueError(f"invalid jersey read for {tracklet_id}") from error
            if frame < 0 or pts < 0 or len(bbox) != 4 or not all(math.isfinite(item) for item in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1] or status not in {"unreviewed", "reviewed", "accepted", "rejected"} or not isinstance(candidates, list):
                raise IdentityCueError(f"invalid jersey read metadata for {tracklet_id}")
            accepted_number_raw = reading.get("accepted_number")
            accepted_number = None if accepted_number_raw is None else int(accepted_number_raw)
            if accepted_number is not None and not 0 <= accepted_number <= 99:
                raise IdentityCueError(f"accepted jersey number for {tracklet_id} must be between zero and 99")
            crop_quality = reading.get("crop_quality", {})
            if not isinstance(crop_quality, dict):
                raise IdentityCueError(f"crop_quality for {tracklet_id} must be an object")
            normalized_quality: dict[str, float] = {}
            for quality_key in ("crop_width_px", "crop_height_px", "bbox_height_px", "detection_score", "laplacian_variance"):
                if crop_quality.get(quality_key) is None:
                    continue
                try:
                    quality_value = float(crop_quality[quality_key])
                except (TypeError, ValueError) as error:
                    raise IdentityCueError(f"invalid {quality_key} crop-quality value for {tracklet_id}") from error
                if not math.isfinite(quality_value) or quality_value < 0 or (quality_key == "detection_score" and quality_value > 1.0):
                    raise IdentityCueError(f"out-of-range {quality_key} crop-quality value for {tracklet_id}")
                normalized_quality[quality_key] = quality_value
            record_key = (tracklet_id, "jersey", frame)
            if record_key in seen_observations:
                raise IdentityCueError(f"duplicate jersey cue for {tracklet_id} at frame {frame}")
            seen_observations.add(record_key)
            normalized: list[dict[str, Any]] = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise IdentityCueError(f"invalid jersey candidate for {tracklet_id}")
                try:
                    number, confidence = int(candidate["number"]), float(candidate["confidence"])
                except (KeyError, TypeError, ValueError) as error:
                    raise IdentityCueError(f"invalid jersey candidate for {tracklet_id}") from error
                if not 0 <= number <= 99 or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise IdentityCueError(f"out-of-range jersey candidate for {tracklet_id}")
                normalized.append({"number": number, "confidence": confidence})
            jersey_reads.append({"source_frame": frame, "source_pts": pts, "bbox_xyxy_px": bbox, "review_status": status, "accepted_number": accepted_number, "model": str(reading.get("model", "unknown")), "crop_quality": normalized_quality, "candidates": sorted(normalized, key=lambda item: (-item["confidence"], item["number"]))})
        embeddings: list[Mapping[str, Any]] = []
        for embedding in raw.get("appearance_embeddings", []):
            if not isinstance(embedding, dict):
                raise IdentityCueError(f"appearance embedding for {tracklet_id} must be an object")
            try:
                frame, pts = int(embedding["source_frame"]), int(embedding["source_pts"])
                bbox = tuple(float(item) for item in embedding["bbox_xyxy_px"])
                vector = tuple(float(item) for item in embedding["vector"])
            except (KeyError, TypeError, ValueError) as error:
                raise IdentityCueError(f"invalid appearance embedding for {tracklet_id}") from error
            model = str(embedding.get("model", "")).strip()
            status = str(embedding.get("review_status", "unreviewed"))
            quality_score = float(embedding.get("crop_quality_score", -1.0))
            if frame < 0 or pts < 0 or len(bbox) != 4 or not all(math.isfinite(item) for item in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1] or not model or status not in {"unreviewed", "reviewed", "accepted", "rejected"} or not vector or not all(math.isfinite(item) for item in vector) or not math.isfinite(quality_score) or not 0.0 <= quality_score <= 1.0:
                raise IdentityCueError(f"invalid appearance embedding metadata for {tracklet_id}")
            record_key = (tracklet_id, "embedding", frame)
            if record_key in seen_observations:
                raise IdentityCueError(f"duplicate appearance embedding for {tracklet_id} at frame {frame}")
            seen_observations.add(record_key)
            embeddings.append({"source_frame": frame, "source_pts": pts, "bbox_xyxy_px": bbox, "model": model, "review_status": status, "crop_quality_score": quality_score, "vector": vector})
        result[tracklet_id] = TrackletCues(tracklet_id, tuple(sorted(jersey_reads, key=lambda item: (item["source_frame"], item["source_pts"]))), tuple(sorted(embeddings, key=lambda item: (item["source_frame"], item["source_pts"], item["model"]))))
    return result


def reviewed_jersey_numbers(cues: Mapping[str, TrackletCues]) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for tracklet_id, tracklet in cues.items():
        counts: dict[int, int] = {}
        for reading in tracklet.jersey_reads:
            if reading.get("review_status") not in {"reviewed", "accepted"}:
                continue
            accepted_number = reading.get("accepted_number")
            if accepted_number is not None:
                number = int(accepted_number)
                counts[number] = counts.get(number, 0) + 1
            else:
                reliable = [candidate for candidate in reading.get("candidates", []) if float(candidate["confidence"]) >= 0.9]
                if len(reliable) == 1:
                    number = int(reliable[0]["number"])
                    counts[number] = counts.get(number, 0) + 1
        result[tracklet_id] = tuple(sorted(number for number, count in counts.items() if count >= 1))
    return result


def _number_distribution(tracklet: TrackletCues | None, *, reviewed_only: bool = False) -> dict[int, float]:
    weights: dict[int, float] = {}
    totals: dict[int, float] = {}
    if tracklet is None:
        return {}
    for reading in tracklet.jersey_reads:
        if reading.get("review_status") == "rejected":
            continue
        if reviewed_only and reading.get("review_status") not in {"reviewed", "accepted"}:
            continue
        if reviewed_only and reading.get("accepted_number") is not None:
            number = int(reading["accepted_number"])
            weights[number] = weights.get(number, 0.0) + 1.0
            totals[number] = totals.get(number, 0.0) + 1.0
            continue
        for candidate in reading.get("candidates", []):
            if reviewed_only and float(candidate["confidence"]) < 0.9:
                continue
            number = int(candidate["number"])
            confidence = float(candidate["confidence"])
            weights[number] = weights.get(number, 0.0) + confidence
            totals[number] = totals.get(number, 0.0) + 1.0
    return {number: weight / totals[number] for number, weight in weights.items() if totals[number]}


def _mean_embeddings(tracklet: TrackletCues | None) -> dict[str, tuple[float, ...]]:
    grouped: dict[str, list[tuple[float, ...]]] = {}
    if tracklet is None:
        return {}
    for item in tracklet.appearance_embeddings:
        if item.get("review_status") == "rejected":
            continue
        if float(item.get("crop_quality_score", 0.0)) < 0.6:
            continue
        vector = tuple(float(value) for value in item["vector"])
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            grouped.setdefault(str(item["model"]), []).append(tuple(value / norm for value in vector))
    result: dict[str, tuple[float, ...]] = {}
    for model, vectors in grouped.items():
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            continue
        mean = tuple(sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension))
        norm = math.sqrt(sum(value * value for value in mean))
        if norm > 0:
            result[model] = tuple(value / norm for value in mean)
    return result


def _median_quality(tracklet: TrackletCues | None) -> dict[str, float | int]:
    readings = list(tracklet.jersey_reads) if tracklet is not None else []
    crop_heights = [float(item.get("crop_quality", {}).get("crop_height_px")) for item in readings if item.get("crop_quality", {}).get("crop_height_px") is not None]
    sharpness = [float(item.get("crop_quality", {}).get("laplacian_variance")) for item in readings if item.get("crop_quality", {}).get("laplacian_variance") is not None]
    box_heights = [float(item.get("crop_quality", {}).get("bbox_height_px")) for item in readings if item.get("crop_quality", {}).get("bbox_height_px") is not None]
    embedding_scores = [float(item.get("crop_quality_score", 0.0)) for item in (tracklet.appearance_embeddings if tracklet is not None else ())]
    def median(values: list[float]) -> float | None:
        ordered = sorted(values)
        if not ordered:
            return None
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    return {
        "jersey_read_count": len(readings),
        "median_crop_height_px": median(crop_heights),
        "median_box_height_px": median(box_heights),
        "median_laplacian_variance": median(sharpness),
        "appearance_embedding_count": len(embedding_scores),
        "appearance_quality_eligible_count": sum(score >= 0.6 for score in embedding_scores),
    }


def validate_cues_against_observations(
    cues: Mapping[str, TrackletCues],
    observations: Sequence[Any],
) -> None:
    """Bind every cue crop to the matching current-run track/frame/PTS/box."""

    indexed = {(row.tracklet_id, row.frame_index): row for row in observations}
    for tracklet_id, tracklet in cues.items():
        records = (*tracklet.jersey_reads, *tracklet.appearance_embeddings)
        for cue in records:
            key = (tracklet_id, int(cue["source_frame"]))
            row = indexed.get(key)
            if row is None:
                raise IdentityCueError(f"identity cue references missing observation {tracklet_id}@{cue['source_frame']}")
            if int(cue["source_pts"]) != row.pts:
                raise IdentityCueError(f"cue PTS does not match current observation {tracklet_id}@{row.frame_index}")
            cue_box = tuple(float(value) for value in cue["bbox_xyxy_px"])
            if len(cue_box) != 4:
                raise IdentityCueError(f"cue box must contain four coordinates for {tracklet_id}@{row.frame_index}")
            if any(abs(first - second) > 0.05 for first, second in zip(cue_box, row.bbox_xyxy_px)):
                raise IdentityCueError(f"cue crop box does not match current observation {tracklet_id}@{row.frame_index}")


def compare_tracklet_cues(left: TrackletCues | None, right: TrackletCues | None) -> dict[str, Any]:
    """Compare optional cues for ranking and reviewed hard-constraint evidence.

    The returned appearance and unreviewed jersey values are proposals only;
    callers must not use them to accept an identity link.
    """

    left_numbers = _number_distribution(left)
    right_numbers = _number_distribution(right)
    common = sorted(set(left_numbers) & set(right_numbers))
    left_reviewed = _number_distribution(left, reviewed_only=True)
    right_reviewed = _number_distribution(right, reviewed_only=True)
    reviewed_common = sorted(set(left_reviewed) & set(right_reviewed))
    reviewed_conflict = bool(left_reviewed and right_reviewed and not reviewed_common)
    jersey_proposal_score = max((min(left_numbers[number], right_numbers[number]) for number in common), default=None)
    left_embeddings = _mean_embeddings(left)
    right_embeddings = _mean_embeddings(right)
    shared_models = sorted(set(left_embeddings) & set(right_embeddings))
    appearance_scores: dict[str, float] = {}
    for model in shared_models:
        first, second = left_embeddings[model], right_embeddings[model]
        if len(first) == len(second):
            appearance_scores[model] = max(-1.0, min(1.0, sum(a * b for a, b in zip(first, second))))
    appearance_similarity = max(appearance_scores.values(), default=None)
    rank_scores: list[tuple[float, float]] = []
    if jersey_proposal_score is not None:
        rank_scores.append((0.25, jersey_proposal_score))
    if appearance_similarity is not None:
        rank_scores.append((0.35, (appearance_similarity + 1.0) / 2.0))
    rank_total = sum(weight for weight, _ in rank_scores)
    review_rank_score = sum(weight * score for weight, score in rank_scores) / rank_total if rank_total else None
    return {
        "reviewed_jersey_conflict": reviewed_conflict,
        "reviewed_jersey_agreement": bool(reviewed_common),
        "reviewed_common_numbers": reviewed_common,
        "jersey_number_proposal_score": jersey_proposal_score,
        "left_jersey_candidates": [{"number": number, "confidence": left_numbers[number]} for number in sorted(left_numbers)],
        "right_jersey_candidates": [{"number": number, "confidence": right_numbers[number]} for number in sorted(right_numbers)],
        "left_crop_quality": _median_quality(left),
        "right_crop_quality": _median_quality(right),
        "appearance_model_similarities": appearance_scores,
        "appearance_similarity_proposal": appearance_similarity,
        "review_rank_score": review_rank_score,
        "decision_use": "review_ranking_only; reviewed_jersey_conflict_is_a_hard_constraint",
    }
