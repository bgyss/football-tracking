"""Conservative, play-scoped resolution of cross-view identity links."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Mapping, Sequence

from .identity import IdentityLink, TeamEvidence, match_tracklets, stable_anonymous_ids
from .identity_cues import TrackletCues, reviewed_jersey_numbers
from .replay import FieldTrack, cross_shot_candidate_evidence


@dataclass(frozen=True, slots=True)
class ResolutionPolicy:
    threshold: float = 0.65
    margin: float = 0.1
    max_field_distance_yards: float = 6.0
    min_overlap_samples: int = 5
    sample_tolerance_s: float = 0.05
    min_overlap_duration_s: float = 0.4
    max_position_uncertainty_yards: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0 or not math.isfinite(self.threshold) or self.margin < 0.0 or not math.isfinite(self.margin):
            raise ValueError("invalid identity resolution thresholds")
        if self.max_field_distance_yards <= 0 or not math.isfinite(self.max_field_distance_yards) or self.min_overlap_samples < 2 or self.sample_tolerance_s <= 0 or not math.isfinite(self.sample_tolerance_s) or self.min_overlap_duration_s < 0 or not math.isfinite(self.min_overlap_duration_s) or self.max_position_uncertainty_yards < 0 or not math.isfinite(self.max_position_uncertainty_yards):
            raise ValueError("invalid identity resolution evidence limits")


@dataclass(frozen=True, slots=True)
class PlayResolution:
    play_id: str
    status: str
    links: tuple[IdentityLink, ...]
    candidate_pairs: int
    shot_ids: tuple[str, ...]
    unresolved_shots: tuple[str, ...]
    rejected_links: tuple[IdentityLink, ...] = ()
    reason: str | None = None
    candidate_scores: tuple[tuple[str, str, float], ...] = ()
    failed_pairs: tuple[tuple[str, str], ...] = ()
    unmatched_tracklets: tuple[str, ...] = ()
    candidate_evidence: tuple[dict[str, object], ...] = ()

    @property
    def accepted_links(self) -> tuple[IdentityLink, ...]:
        return tuple(link for link in self.links if link.decision == "same")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "play_id": self.play_id,
            "status": self.status,
            "reason": self.reason,
            "shot_ids": list(self.shot_ids),
            "unresolved_shots": list(self.unresolved_shots),
            "candidate_pairs": self.candidate_pairs,
            "candidate_scores": [{"left_key": left, "right_key": right, "score": score} for left, right, score in self.candidate_scores],
            "candidate_evidence": [dict(record) for record in self.candidate_evidence],
            "failed_pairs": [{"left_shot": left, "right_shot": right} for left, right in self.failed_pairs],
            "unmatched_tracklets": list(self.unmatched_tracklets),
            "accepted_links": len(self.accepted_links),
            "links": [
                {
                    "left_key": link.left_key,
                    "right_key": link.right_key,
                    "decision": link.decision,
                    "score": link.score,
                    "alternative_score": link.alternative_score,
                    "ambiguity_margin": link.ambiguity_margin,
                    "rejection_reason": link.rejection_reason,
                    "evidence_keys": list(link.evidence_keys),
                }
                for link in self.links + self.rejected_links
            ],
        }


def _safe_component_links(
    tracklet_ids: Sequence[str],
    links: Sequence[IdentityLink],
    tracklet_teams: Mapping[str, str] | None = None,
    tracklet_frames: Mapping[str, Sequence[int]] | None = None,
    tracklet_play_ids: Mapping[str, str | None] | None = None,
    tracklet_jerseys: Mapping[str, Sequence[int]] | None = None,
) -> tuple[list[IdentityLink], list[IdentityLink]]:
    accepted: list[IdentityLink] = []
    rejected: list[IdentityLink] = []
    for link in sorted((item for item in links if item.decision == "same"), key=lambda item: (item.left_key, item.right_key or "")):
        candidate = accepted + [link]
        try:
            stable_anonymous_ids(
                tracklet_ids,
                candidate,
                tracklet_frames=tracklet_frames,
                tracklet_play_ids=tracklet_play_ids,
                tracklet_teams=tracklet_teams,
                tracklet_jerseys=tracklet_jerseys,
            )
        except ValueError:
            rejected.append(IdentityLink(link.left_key, link.right_key, "insufficient_evidence", link.score, link.evidence_keys, link.alternative_score, link.ambiguity_margin, "component_conflict"))
        else:
            accepted.append(link)
    return accepted, rejected


def resolve_play_identities(
    play: Mapping[str, Any] | Any,
    segments: Mapping[str, Sequence[FieldTrack]],
    calibration: Any = None,
    time_map: Any = None,
    teams: Mapping[str, TeamEvidence] | None = None,
    policy: ResolutionPolicy | None = None,
    tracklet_frames: Mapping[str, Sequence[int]] | None = None,
    tracklet_play_ids: Mapping[str, str | None] | None = None,
    tracklet_jerseys: Mapping[str, Sequence[int]] | None = None,
    tracklet_cues: Mapping[str, TrackletCues] | None = None,
) -> PlayResolution:
    """Resolve all shot pairs in one reviewed play.

    ``segments`` must already contain calibrated, play-time samples. The
    calibration and time_map parameters are retained in the public interface
    for callers that need to attach their provenance; this function never uses
    raw image coordinates or silently extrapolates either input. When supplied,
    ``tracklet_frames`` allows component checks to distinguish reviewed,
    non-overlapping fragments from simultaneous same-shot tracks.
    """

    del calibration, time_map
    selected_policy = policy or ResolutionPolicy()
    play_id = str(play.get("play_id", "")) if isinstance(play, Mapping) else str(getattr(play, "play_id", ""))
    if not play_id:
        raise ValueError("play must have a play_id")
    shot_ids = tuple(sorted(str(shot_id) for shot_id in (play.get("shot_ids", segments.keys()) if isinstance(play, Mapping) else getattr(play, "shots", lambda: tuple(segments))())))
    teams = teams or {}
    present = [shot_id for shot_id in shot_ids if segments.get(shot_id)]
    unresolved = [shot_id for shot_id in shot_ids if not segments.get(shot_id)]
    if len(present) < 2:
        unmatched = tuple(sorted(track.tracklet_id for shot_id in present for track in segments[shot_id]))
        return PlayResolution(play_id, "abstained", (), 0, shot_ids, tuple(unresolved or shot_ids), reason="fewer than two shots have calibrated field tracks", unmatched_tracklets=unmatched)
    all_links: list[IdentityLink] = []
    candidate_count = 0
    candidate_records: list[tuple[str, str, float]] = []
    candidate_evidence_records: list[dict[str, object]] = []
    failed_pairs: list[tuple[str, str]] = []
    for left_shot, right_shot in combinations(present, 2):
        left = list(segments[left_shot])
        right = list(segments[right_shot])
        evidence = cross_shot_candidate_evidence(
            left,
            right,
            teams,
            max_field_distance_yards=selected_policy.max_field_distance_yards,
            min_overlap_samples=selected_policy.min_overlap_samples,
            sample_tolerance_s=selected_policy.sample_tolerance_s,
            min_overlap_duration_s=selected_policy.min_overlap_duration_s,
            max_position_uncertainty_yards=selected_policy.max_position_uncertainty_yards,
            tracklet_cues=tracklet_cues,
        )
        candidates = {key: float(record["score"]) for key, record in evidence.items() if record["eligible"] and record["score"] is not None}
        candidate_count += len(candidates)
        candidate_records.extend((left_key, right_key, float(score)) for (left_key, right_key), score in sorted(candidates.items()))
        candidate_evidence_records.extend(
            {**record, "pair_left_shot": left_shot, "pair_right_shot": right_shot}
            for _, record in sorted(evidence.items())
        )
        pair_links = match_tracklets(
            [track.tracklet_id for track in left],
            [track.tracklet_id for track in right],
            candidates,
            threshold=selected_policy.threshold,
            margin=selected_policy.margin,
        )
        enriched_links: list[IdentityLink] = []
        for link in pair_links:
            keys: tuple[str, ...] = ()
            if link.right_key is not None:
                keys = ("team_agreement", "calibrated_field_position", "trajectory_shape")
                cue_record = evidence.get((link.left_key, link.right_key), {}).get("identity_cues", {})
                if isinstance(cue_record, Mapping) and cue_record.get("reviewed_jersey_agreement"):
                    keys += ("reviewed_jersey_agreement",)
            enriched_links.append(replace(link, evidence_keys=keys))
        pair_links = enriched_links
        all_links.extend(pair_links)
        if not any(link.decision == "same" for link in pair_links):
            failed_pairs.append((left_shot, right_shot))
        # A present shot with no accepted pair is represented by its
        # ``insufficient_evidence`` links. ``unresolved_shots`` is reserved for
        # shots that could not participate at all (for example missing
        # calibration or no valid field samples), so a two-shot abstention
        # remains distinguishable from a missing view.
    tracklet_ids = [track.tracklet_id for shot_id in present for track in segments[shot_id]]
    tracklet_teams = {tracklet_id: evidence.team for tracklet_id, evidence in teams.items()}
    if tracklet_jerseys is None and tracklet_cues is not None:
        tracklet_jerseys = reviewed_jersey_numbers(tracklet_cues)
    accepted, rejected = _safe_component_links(
        tracklet_ids,
        all_links,
        tracklet_teams,
        tracklet_frames,
        tracklet_play_ids,
        tracklet_jerseys,
    )
    # Keep one deterministic link record per left/right pair, preferring an
    # accepted edge and then the highest score.
    chosen: dict[tuple[str, str | None], IdentityLink] = {}
    for link in all_links:
        key = (link.left_key, link.right_key)
        previous = chosen.get(key)
        if previous is None or (link.decision == "same", link.score) > (previous.decision == "same", previous.score):
            chosen[key] = link
    final_links = [link for link in sorted(chosen.values(), key=lambda item: (item.left_key, item.right_key or "")) if link in accepted or link.decision != "same"]
    if len(present) > 2:
        for left_shot, right_shot in failed_pairs:
            unresolved.extend((left_shot, right_shot))
    unresolved_shots = tuple(sorted(set(unresolved)))
    linked_tracklets = {link.left_key for link in accepted} | {link.right_key for link in accepted if link.right_key is not None}
    unmatched_tracklets = tuple(sorted(set(tracklet_ids) - linked_tracklets))
    if accepted and not unresolved_shots and not rejected and not unmatched_tracklets:
        status, reason = "resolved", "all present shot pairs cleared the reviewed policy"
    elif accepted:
        status, reason = "partially_resolved", "some aligned shots or segments lacked an eligible link"
    else:
        status, reason = "abstained", "no candidate pair cleared the reviewed policy"
    return PlayResolution(play_id, status, tuple(final_links), candidate_count, shot_ids, unresolved_shots, tuple(rejected), reason, tuple(candidate_records), tuple(sorted(failed_pairs)), unmatched_tracklets, tuple(candidate_evidence_records))
