"""Team evidence and constrained, anonymous identity linking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .tracking import TrackObservation


@dataclass(frozen=True, slots=True)
class TeamEvidence:
    team: str
    score: float
    source: str
    crop_count: int
    assignment_coverage: float = 0.0
    ambiguity: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("team score must be between 0 and 1")
        if self.crop_count < 0:
            raise ValueError("crop_count must be non-negative")
        if not 0.0 <= self.assignment_coverage <= 1.0:
            raise ValueError("assignment coverage must be between 0 and 1")
        if not 0.0 <= self.ambiguity <= 1.0:
            raise ValueError("team ambiguity must be between 0 and 1")

    @property
    def eligible(self) -> bool:
        return self.team != "unknown" and self.score > 0.0 and self.ambiguity < 0.5


@dataclass(frozen=True, slots=True)
class TrackletSummary:
    tracklet_id: str
    observations: tuple[TrackObservation, ...] = ()
    team_features: tuple[tuple[float, float, float], ...] = ()
    jersey_candidates: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for feature in self.team_features:
            if len(feature) != 3 or any(not 0.0 <= float(value) <= 1.0 for value in feature):
                raise ValueError("team features must contain RGB values between 0 and 1")


@dataclass(frozen=True, slots=True)
class IdentityLink:
    left_key: str
    right_key: str | None
    decision: str
    score: float
    evidence_keys: tuple[str, ...] = ()
    alternative_score: float | None = None
    ambiguity_margin: float | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if self.decision not in {"same", "different", "insufficient_evidence"}:
            raise ValueError("unsupported identity decision")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("identity score must be between 0 and 1")
        if self.alternative_score is not None and not 0.0 <= self.alternative_score <= 1.0:
            raise ValueError("alternative score must be between 0 and 1")
        if self.ambiguity_margin is not None and self.ambiguity_margin < 0.0:
            raise ValueError("ambiguity margin must be non-negative")
        if self.decision == "same" and self.right_key is None:
            raise ValueError("same links require a right key")


def team_feature_from_crop(crop_bgr: np.ndarray) -> tuple[float, float, float] | None:
    """Return a robust torso RGB mean, or None for an unusable crop."""

    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3 or crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return None
    height, width = crop_bgr.shape[:2]
    torso = crop_bgr[int(height * 0.2) : int(height * 0.8), int(width * 0.2) : int(width * 0.8)]
    if torso.size == 0:
        return None
    rgb = torso[:, :, ::-1].astype(np.float32) / 255.0
    pixels = rgb.reshape(-1, 3)
    # A trimmed mean reduces turf/skin/background contamination in a crop.
    values = np.percentile(pixels, [20, 80], axis=0)
    trimmed = pixels[(pixels >= values[0]).all(axis=1) & (pixels <= values[1]).all(axis=1)]
    mean = np.mean(trimmed if len(trimmed) else pixels, axis=0)
    return tuple(float(value) for value in mean)


def _cluster_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(features) == 1:
        return features.copy(), np.zeros(1, dtype=int)
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    first = 0
    second = int(np.unravel_index(np.argmax(distances), distances.shape)[1])
    if second == first:
        second = 1
    centers = np.vstack([features[first], features[second]])
    labels = np.zeros(len(features), dtype=int)
    for _ in range(12):
        labels = np.argmin(np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2), axis=1)
        next_centers = np.vstack([
            features[labels == label].mean(axis=0) if np.any(labels == label) else centers[label]
            for label in range(2)
        ])
        if np.allclose(next_centers, centers):
            break
        centers = next_centers
    return centers, labels


def resolve_teams(
    tracklets: Sequence[TrackletSummary],
    prototypes: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, TeamEvidence]:
    """Aggregate crop evidence into team labels, retaining unknown cases."""

    result: dict[str, TeamEvidence] = {}
    if prototypes:
        names = sorted(prototypes)
        centers = np.asarray([prototypes[name] for name in names], dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("team prototypes must be RGB triples")
        for tracklet in tracklets:
            if not tracklet.team_features:
                result[tracklet.tracklet_id] = TeamEvidence("unknown", 0.0, "no_crops", 0)
                continue
            features = np.asarray(tracklet.team_features, dtype=float)
            distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
            winners = np.argmin(distances, axis=1)
            counts = np.bincount(winners, minlength=len(names))
            winner = int(np.argmax(counts))
            ordered_counts = sorted((int(count), index) for index, count in enumerate(counts))
            runner_count = ordered_counts[-2][0] if len(ordered_counts) > 1 else 0
            coverage = float(counts[winner] / len(winners)) if len(winners) else 0.0
            ambiguity = float(runner_count / len(winners)) if len(winners) else 1.0
            assigned = distances[winners == winner, winner]
            score = float(np.mean(1.0 / (1.0 + assigned * 4.0))) if len(assigned) else 0.0
            ranked_distances = np.sort(distances, axis=1)
            distance_ambiguity = float(np.mean((ranked_distances[:, 1] - ranked_distances[:, 0]) < 0.05)) if distances.shape[1] > 1 else 0.0
            effective_ambiguity = max(ambiguity, distance_ambiguity)
            if effective_ambiguity >= 0.5 or score < 0.5:
                result[tracklet.tracklet_id] = TeamEvidence("unknown", score, "ambiguous_rgb_prototype", len(tracklet.team_features), coverage, effective_ambiguity)
            else:
                result[tracklet.tracklet_id] = TeamEvidence(names[winner], score, "rgb_prototype", len(tracklet.team_features), coverage, effective_ambiguity)
        return result
    all_features = [feature for tracklet in tracklets for feature in tracklet.team_features]
    if not all_features:
        return {tracklet.tracklet_id: TeamEvidence("unknown", 0.0, "no_crops", 0) for tracklet in tracklets}
    centers, _ = _cluster_features(np.asarray(all_features, dtype=float))
    for tracklet in tracklets:
        if not tracklet.team_features:
            result[tracklet.tracklet_id] = TeamEvidence("unknown", 0.0, "no_crops", 0)
            continue
        features = np.asarray(tracklet.team_features, dtype=float)
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        winner = int(np.bincount(labels, minlength=2).argmax())
        assigned = distances[labels == winner, winner]
        result[tracklet.tracklet_id] = TeamEvidence(f"team_{winner}", float(np.mean(1.0 / (1.0 + assigned * 4.0))), "rgb_cluster", len(tracklet.team_features), float(np.mean(labels == winner)), 0.0)
    return result


def match_tracklets(
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    candidate_scores: Mapping[tuple[str, str], float],
    threshold: float = 0.65,
    margin: float = 0.1,
) -> list[IdentityLink]:
    """Globally match tracklets once, and abstain where evidence is weak."""

    if not 0.0 <= threshold <= 1.0 or margin < 0.0:
        raise ValueError("invalid matching thresholds")
    left = sorted(set(left_keys))
    right = sorted(set(right_keys))
    if not left or not right:
        return [IdentityLink(key, None, "insufficient_evidence", 0.0) for key in left]
    matrix = np.full((len(left), len(right)), -1.0, dtype=float)
    for i, left_key in enumerate(left):
        for j, right_key in enumerate(right):
            score = candidate_scores.get((left_key, right_key))
            if score is not None:
                if not 0.0 <= score <= 1.0:
                    raise ValueError("candidate scores must be between 0 and 1")
                matrix[i, j] = float(score)
    # Keep unmatched as an explicit option.  Invalid and below-threshold edges
    # are absent, while a dummy column has score zero.  This means an isolated
    # high score can be selected without forcing an unrelated low score into a
    # one-to-one assignment.
    eligible = np.where(matrix >= threshold, matrix, -1.0)
    augmented = np.concatenate([eligible, np.zeros((len(left), len(left)), dtype=float)], axis=1)

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise RuntimeError("scipy is required for deterministic identity assignment") from error

    def solve(values: np.ndarray) -> tuple[dict[int, int], float]:
        rows, columns = linear_sum_assignment(-values)
        assignment = {int(row): int(column) for row, column in zip(rows, columns)}
        return assignment, float(sum(values[row, column] for row, column in assignment.items()))

    assignment, baseline_objective = solve(augmented)
    links: list[IdentityLink] = []
    for row, left_key in enumerate(left):
        column = assignment.get(row)
        if column is None or column >= len(right) or eligible[row, column] < 0:
            links.append(IdentityLink(left_key, None, "insufficient_evidence", 0.0))
            continue
        score = float(eligible[row, column])
        # A genuine ambiguity is measured against the best complete assignment
        # after forbidding this edge.  Do not remove already assigned columns
        # from the alternative: a competing row may be the reason this edge is
        # unsafe.  The dummy columns remain available in every alternative.
        without_edge = augmented.copy()
        without_edge[row, column] = 0.0
        _, alternative_objective = solve(without_edge)
        # Convert the global objective delta into the edge's score scale.  The
        # objective can contain many rows, so compare the best competing edge
        # contribution by subtracting all unchanged selected contributions.
        global_margin = max(0.0, baseline_objective - alternative_objective)
        alternative_score = max(0.0, min(1.0, score - global_margin))
        if global_margin < margin:
            links.append(IdentityLink(
                left_key,
                right[column],
                "insufficient_evidence",
                score,
                alternative_score=alternative_score,
                ambiguity_margin=global_margin,
                rejection_reason="global_assignment_ambiguous",
            ))
        else:
            links.append(IdentityLink(
                left_key,
                right[column],
                "same",
                score,
                alternative_score=alternative_score,
                ambiguity_margin=global_margin,
            ))
    return links


def stable_anonymous_ids(tracklet_ids: Sequence[str], links: Sequence[IdentityLink]) -> dict[str, str]:
    parent = {key: key for key in sorted(set(tracklet_ids))}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        if left not in parent or right not in parent:
            raise ValueError("identity link references an unknown tracklet")
        first, second = find(left), find(right)
        if first != second:
            first_shots = {member.split(":", 1)[0] for member in parent if find(member) == first}
            second_shots = {member.split(":", 1)[0] for member in parent if find(member) == second}
            if first_shots & second_shots:
                raise ValueError("identity link would merge two tracklets from the same shot")
            parent[max(first, second)] = min(first, second)

    for link in links:
        if link.decision == "same" and link.right_key is not None:
            union(link.left_key, link.right_key)
    components: dict[str, list[str]] = {}
    for key in parent:
        components.setdefault(find(key), []).append(key)
    ordered = sorted(components.values(), key=lambda members: min(members))
    result: dict[str, str] = {}
    for index, members in enumerate(ordered, start=1):
        for member in members:
            result[member] = f"P{index:02d}"
    return result
