from __future__ import annotations

import pytest
import numpy as np

from football_tracking.identity import (
    IdentityLink,
    TrackletSummary,
    match_tracklets,
    resolve_teams,
    stable_anonymous_ids,
)


def test_resolve_teams_uses_prototypes_and_preserves_unknown() -> None:
    tracklets = [
        TrackletSummary("a", team_features=((0.1, 0.2, 0.9), (0.12, 0.2, 0.86))),
        TrackletSummary("b", team_features=((0.9, 0.85, 0.2),)),
        TrackletSummary("official"),
    ]

    result = resolve_teams(
        tracklets,
        prototypes={"DET": (0.1, 0.2, 0.9), "LAR": (0.9, 0.85, 0.2)},
    )

    assert result["a"].team == "DET"
    assert result["a"].score > 0.8
    assert result["b"].team == "LAR"
    assert result["official"].team == "unknown"
    assert result["official"].source == "no_crops"


def test_resolve_teams_keeps_tied_crop_votes_unknown() -> None:
    result = resolve_teams(
        [TrackletSummary("ambiguous", team_features=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))],
        prototypes={"DET": (0.0, 0.0, 0.0), "LAR": (1.0, 1.0, 1.0)},
    )
    assert result["ambiguous"].team == "unknown"
    assert result["ambiguous"].source == "ambiguous_rgb_prototype"


def test_match_tracklets_is_one_to_one_and_abstains_on_low_margin() -> None:
    scores = {
        ("left-1", "right-1"): 0.95,
        ("left-1", "right-2"): 0.40,
        ("left-2", "right-1"): 0.93,
        ("left-2", "right-2"): 0.92,
    }

    links = match_tracklets(["left-1", "left-2"], ["right-1", "right-2"], scores, threshold=0.9, margin=0.02)

    accepted = {(link.left_key, link.right_key) for link in links if link.decision == "same"}
    assert accepted == {("left-1", "right-1"), ("left-2", "right-2")}
    assert len({link.left_key for link in links if link.decision == "same"}) == 2

    abstained = match_tracklets(["a"], ["b", "c"], {("a", "b"): 0.91, ("a", "c"): 0.90}, threshold=0.9, margin=0.02)
    assert abstained[0].decision == "insufficient_evidence"


def test_match_tracklets_abstains_on_globally_tied_assignment() -> None:
    scores = {(left, right): 0.9 for left in ("a", "b") for right in ("c", "d")}
    links = match_tracklets(["a", "b"], ["c", "d"], scores)
    assert all(link.decision == "insufficient_evidence" for link in links)
    assert all(link.rejection_reason == "global_assignment_ambiguous" for link in links)


def test_stable_anonymous_ids_rejects_same_shot_component_conflict() -> None:
    with pytest.raises(ValueError, match="same shot"):
        stable_anonymous_ids(["shot-0:a", "shot-0:b", "shot-1:c"], [IdentityLink("shot-0:a", "shot-1:c", "same", 0.9), IdentityLink("shot-0:b", "shot-1:c", "same", 0.9)])


def test_stable_anonymous_ids_are_deterministic_and_validate_links() -> None:
    links = [IdentityLink("shot-1:t2", "shot-0:t9", "same", 0.93, ("crop-1",))]

    first = stable_anonymous_ids(["shot-0:t9", "shot-1:t2", "shot-2:t1"], links)
    second = stable_anonymous_ids(["shot-2:t1", "shot-1:t2", "shot-0:t9"], links)

    assert first == second
    assert first["shot-0:t9"] == first["shot-1:t2"]
    assert first["shot-2:t1"] != first["shot-0:t9"]


def test_team_feature_from_crop_returns_rgb_mean_for_image_crop() -> None:
    from football_tracking.identity import team_feature_from_crop

    crop = np.zeros((10, 12, 3), dtype=np.uint8)
    crop[:, :, 0] = 30  # B
    crop[:, :, 1] = 60  # G
    crop[:, :, 2] = 240  # R

    feature = team_feature_from_crop(crop)

    assert feature is not None
    assert feature == pytest.approx((240 / 255, 60 / 255, 30 / 255), abs=0.01)
