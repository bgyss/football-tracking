from __future__ import annotations

import json

import pytest

from football_tracking.replay import (
    PlayAlignment,
    PlayAnchor,
    PlayTimeCorrespondence,
    PlayTimeMap,
    ReplayAlignmentError,
    load_play_alignment,
)


def test_play_time_is_anchor_relative_and_shot_scoped() -> None:
    alignment = PlayAlignment("play-1", (PlayAnchor("shot-0", 120, "snap"), PlayAnchor("shot-1", 820, "snap")))

    assert alignment.shots() == ("shot-0", "shot-1")
    assert alignment.play_time_s("shot-0", 120, 60.0) == pytest.approx(0.0)
    assert alignment.play_time_s("shot-1", 820, 60.0) == pytest.approx(0.0)
    assert alignment.play_time_s("shot-0", 150, 60.0) == pytest.approx(0.5)
    assert alignment.play_time_s("shot-1", 850, 60.0) == pytest.approx(0.5)
    assert alignment.play_time_s("shot-2", 10, 60.0) is None


def test_alignment_requires_review_two_shots_and_positive_fps(tmp_path) -> None:
    unreviewed = tmp_path / "unreviewed.json"
    unreviewed.write_text(json.dumps({"play_id": "play-1", "anchors": []}), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError):
        load_play_alignment(unreviewed)

    single = tmp_path / "single.json"
    single.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [{"shot_id": "shot-0", "source_frame": 120, "event": "snap"}],
    }), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError):
        load_play_alignment(single)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 120, "event": "snap"},
            {"shot_id": "shot-0", "source_frame": 130, "event": "snap"},
        ],
    }), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError):
        load_play_alignment(duplicate)

    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 120, "event": "snap"},
            {"shot_id": "shot-1", "source_frame": 820, "event": "snap"},
        ],
    }), encoding="utf-8")
    alignment = load_play_alignment(good)
    assert alignment.play_id == "play-1"
    assert alignment.shots() == ("shot-0", "shot-1")

    with pytest.raises(ValueError):
        alignment.play_time_s("shot-0", 120, 0.0)


def test_load_play_alignment_rejects_malformed_anchors(tmp_path) -> None:
    # Test negative source_frame
    negative_frame = tmp_path / "negative_frame.json"
    negative_frame.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": -1, "event": "snap"},
            {"shot_id": "shot-1", "source_frame": 820, "event": "snap"},
        ],
    }), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError):
        load_play_alignment(negative_frame)

    # Test non-numeric source_frame
    non_numeric_frame = tmp_path / "non_numeric_frame.json"
    non_numeric_frame.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": "abc", "event": "snap"},
            {"shot_id": "shot-1", "source_frame": 820, "event": "snap"},
        ],
    }), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError):
        load_play_alignment(non_numeric_frame)

    # Test empty event string
    empty_event = tmp_path / "empty_event.json"
    empty_event.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 120, "event": ""},
            {"shot_id": "shot-1", "source_frame": 820, "event": "snap"},
        ],
    }), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError):
        load_play_alignment(empty_event)


def test_play_time_map_uses_pts_without_extrapolation() -> None:
    mapping = PlayTimeMap([
        PlayTimeCorrespondence("shot-0", 100, 0.0, "snap"),
        PlayTimeCorrespondence("shot-0", 200, 1.0, "contact"),
        PlayTimeCorrespondence("shot-1", 1000, 0.0, "snap"),
        PlayTimeCorrespondence("shot-1", 1100, 1.0, "contact"),
    ])
    assert mapping.at("shot-0", 150) == pytest.approx(0.5)
    assert mapping.at("shot-0", 99) is None
    assert mapping.at("shot-2", 150) is None
    check = PlayTimeCorrespondence("shot-0", 175, 0.75, "release")
    assert mapping.validate([check])["gate"] is True


def test_play_time_map_rejects_nonmonotonic_correspondences() -> None:
    with pytest.raises(ReplayAlignmentError):
        PlayTimeMap([
            PlayTimeCorrespondence("shot-0", 100, 0.0, "snap"),
            PlayTimeCorrespondence("shot-0", 200, 0.0, "contact"),
        ])


def test_alignment_loader_accepts_reviewed_pts_correspondences(tmp_path) -> None:
    path = tmp_path / "pts-alignment.json"
    path.write_text(json.dumps({
        "reviewed": True,
        "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 0, "source_pts": 100, "play_time_s": 0.0, "event": "snap"},
            {"shot_id": "shot-1", "source_frame": 10, "source_pts": 1000, "play_time_s": 0.0, "event": "snap"},
        ],
        "correspondences": [
            {"shot_id": "shot-0", "source_pts": 200, "play_time_s": 1.0, "event": "contact"},
            {"shot_id": "shot-1", "source_pts": 1100, "play_time_s": 1.0, "event": "contact"},
        ],
    }), encoding="utf-8")
    alignment = load_play_alignment(path)
    assert alignment.time_map is not None
    assert alignment.play_time_at_pts("shot-0", 150, (1, 100), fps=100, frame_index=5) == pytest.approx(0.5)


from football_tracking.identity import TeamEvidence
from football_tracking.replay import FieldTrack, cross_shot_candidate_scores


def line(tracklet_id, shot_id, x0, y0, *, dx=1.0, count=10, start=0.0, step=0.1):
    return FieldTrack(
        tracklet_id, shot_id,
        tuple((start + index * step, x0 + index * dx, y0) for index in range(count)),
    )


def teams(**assignment):
    return {key: TeamEvidence(value, 0.95, "rgb_prototype", 20) for key, value in assignment.items()}


def test_matching_field_motion_scores_high_and_distant_motion_is_dropped() -> None:
    left = [line("shot-0:t1", "shot-0", 10.0, 20.0), line("shot-0:t2", "shot-0", 60.0, 40.0)]
    right = [line("shot-1:t1", "shot-1", 10.0, 20.0), line("shot-1:t2", "shot-1", 60.0, 40.0)]
    evidence = teams(**{
        "shot-0:t1": "DET", "shot-0:t2": "LAR", "shot-1:t1": "DET", "shot-1:t2": "LAR",
    })

    scores = cross_shot_candidate_scores(left, right, evidence)

    assert scores[("shot-0:t1", "shot-1:t1")] > 0.9
    assert scores[("shot-0:t2", "shot-1:t2")] > 0.9
    # 50 yards apart, well beyond max_field_distance_yards: no candidate at all.
    assert ("shot-0:t1", "shot-1:t2") not in scores


def test_team_conflict_and_thin_overlap_produce_no_candidate() -> None:
    left = [line("shot-0:t1", "shot-0", 10.0, 20.0)]
    right = [line("shot-1:t1", "shot-1", 10.0, 20.0)]

    conflicting = teams(**{"shot-0:t1": "DET", "shot-1:t1": "LAR"})
    assert cross_shot_candidate_scores(left, right, conflicting) == {}

    unknown = teams(**{"shot-0:t1": "unknown", "shot-1:t1": "DET"})
    assert cross_shot_candidate_scores(left, right, unknown) == {}

    thin_left = [line("shot-0:t1", "shot-0", 10.0, 20.0, count=3)]
    thin_right = [line("shot-1:t1", "shot-1", 10.0, 20.0, count=3)]
    agreeing = teams(**{"shot-0:t1": "DET", "shot-1:t1": "DET"})
    assert cross_shot_candidate_scores(thin_left, thin_right, agreeing) == {}


def test_opposing_trajectory_shape_scores_below_matching_shape() -> None:
    left = [line("shot-0:t1", "shot-0", 10.0, 20.0, dx=1.0)]
    right = [
        line("shot-1:same", "shot-1", 10.0, 20.0, dx=1.0),
        line("shot-1:reverse", "shot-1", 10.0, 20.0, dx=-1.0),
    ]
    evidence = teams(**{"shot-0:t1": "DET", "shot-1:same": "DET", "shot-1:reverse": "DET"})

    # The reversed track ends up a mean 9.0 yards away, past the default gate,
    # so by the hard-constraint rule it is not a candidate at all.
    default_gate = cross_shot_candidate_scores(left, right, evidence)
    assert ("shot-0:t1", "shot-1:reverse") not in default_gate

    # Widen the gate so both pairs are scored, and the shape term is what separates them.
    scores = cross_shot_candidate_scores(left, right, evidence, max_field_distance_yards=20.0)
    assert scores[("shot-0:t1", "shot-1:same")] > scores[("shot-0:t1", "shot-1:reverse")]


def test_scores_are_bounded_and_deterministic() -> None:
    left = [line("shot-0:t1", "shot-0", 10.0, 20.0)]
    right = [line("shot-1:t1", "shot-1", 10.4, 20.3)]
    evidence = teams(**{"shot-0:t1": "DET", "shot-1:t1": "DET"})

    first = cross_shot_candidate_scores(left, right, evidence)
    second = cross_shot_candidate_scores(left, right, evidence)

    assert first == second
    assert all(0.0 <= value <= 1.0 for value in first.values())


def test_paired_samples_enforces_exclusivity_of_right_matches() -> None:
    # Left track: six samples clustered around time 0.0 (start=0.0, step=0.01),
    # giving times 0.00, 0.01, 0.02, 0.03, 0.04, 0.05.
    # Right track: one sample at time 0.0.
    # With sample_tolerance_s=0.05, all six left samples are within tolerance.
    # Without exclusivity, all six left samples claim the right sample,
    # yielding six pairs, which passes the min_overlap_samples=5 floor.
    # With exclusivity, only one left sample claims the right sample, yielding
    # one pair, which fails the min_overlap_samples gate.
    # This fixture discriminates: exclusive and non-exclusive pairing produce
    # opposite outcomes relative to the floor.
    left = [line("shot-0:t1", "shot-0", 10.0, 20.0, count=6, start=0.0, step=0.01)]
    right = [line("shot-1:t1", "shot-1", 10.0, 20.0, count=1, start=0.0, step=0.1)]
    evidence = teams(**{"shot-0:t1": "DET", "shot-1:t1": "DET"})

    # With default min_overlap_samples=5, exclusive pairing yields one pair and
    # fails the gate; non-exclusive would yield six and pass. A regression
    # reintroducing sample reuse would make this test fail.
    scores = cross_shot_candidate_scores(left, right, evidence)
    assert ("shot-0:t1", "shot-1:t1") not in scores
