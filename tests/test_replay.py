from __future__ import annotations

import json

import pytest

from football_tracking.replay import (
    PlayAlignment,
    PlayAlignmentSet,
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
    pts_alignment = PlayAlignment("play-2", (PlayAnchor("shot-0", 120, "snap", 1000, 0.25), PlayAnchor("shot-1", 820, "snap", 2000, 0.25)))
    assert pts_alignment.play_time_at_pts("shot-0", 1100, (1, 1000)) == pytest.approx(0.35)


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


def test_play_time_map_rejects_mismatched_events_between_shots() -> None:
    with pytest.raises(ReplayAlignmentError, match="same reviewed events"):
        PlayTimeMap([
            PlayTimeCorrespondence("shot-0", 100, 0.0, "snap"),
            PlayTimeCorrespondence("shot-0", 200, 1.0, "contact"),
            PlayTimeCorrespondence("shot-1", 1000, 0.0, "snap"),
            PlayTimeCorrespondence("shot-1", 1100, 1.0, "release"),
        ])


def test_fitted_play_time_map_does_not_extrapolate_through_gaps() -> None:
    mapping = PlayTimeMap([
        PlayTimeCorrespondence("shot-0", 100, 0.0, "snap"),
        PlayTimeCorrespondence("shot-0", 200, 1.0, "contact"),
        PlayTimeCorrespondence("shot-1", 1000, 0.0, "snap"),
        PlayTimeCorrespondence("shot-1", 1100, 1.0, "contact"),
    ])
    alignment = PlayAlignment("play-1", (PlayAnchor("shot-0", 0, "snap"), PlayAnchor("shot-1", 1, "snap")), mapping)
    assert alignment.play_time_at_pts("shot-0", 50, (1, 100)) is None


def test_play_time_map_rejects_nonmonotonic_correspondences() -> None:
    with pytest.raises(ReplayAlignmentError):
        PlayTimeMap([
            PlayTimeCorrespondence("shot-0", 100, 0.0, "snap"),
            PlayTimeCorrespondence("shot-0", 200, 0.0, "contact"),
        ])


def test_field_track_rejects_unsorted_or_nonfinite_samples() -> None:
    with pytest.raises(ValueError, match="ordered"):
        FieldTrack("t", "shot-0", ((1.0, 0.0, 0.0), (0.5, 0.0, 0.0)))
    with pytest.raises(ValueError, match="finite"):
        FieldTrack("t", "shot-0", ((0.0, float("nan"), 0.0), (1.0, 0.0, 0.0)))


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
    assert alignment.timing_eligible is False
    validated = json.loads(path.read_text(encoding="utf-8"))
    validated["validation_correspondences"] = [{"shot_id": "shot-0", "source_pts": 150, "play_time_s": 0.5, "event": "release"}, {"shot_id": "shot-1", "source_pts": 1050, "play_time_s": 0.5, "event": "release"}]
    path.write_text(json.dumps(validated), encoding="utf-8")
    assert load_play_alignment(path).timing_eligible is True


def test_multi_play_alignment_loader_sorts_and_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "alignments.json"
    anchor = lambda shot, frame: {"shot_id": shot, "source_frame": frame, "event": "snap"}
    path.write_text(json.dumps({"reviewed": True, "plays": [
        {"play_id": "p2", "anchors": [anchor("shot-2", 20), anchor("shot-3", 30)]},
        {"play_id": "p1", "anchors": [anchor("shot-0", 0), anchor("shot-1", 10)]},
    ]}), encoding="utf-8")
    from football_tracking.replay import load_play_alignments
    loaded = load_play_alignments(path)
    assert isinstance(loaded, PlayAlignmentSet)
    assert [play.play_id for play in loaded.plays] == ["p1", "p2"]
    serialized = loaded.to_dict()
    assert serialized["plays"][0]["play_id"] == "p1"
    with pytest.raises(ReplayAlignmentError, match="duplicate play_id"):
        PlayAlignmentSet((loaded.plays[0], loaded.plays[0]))


def test_alignment_loader_rejects_declared_source_hash_mismatch(tmp_path) -> None:
    path = tmp_path / "bad-source.json"
    path.write_text(json.dumps({"reviewed": True, "source_sha256": "expected", "play_id": "p1", "anchors": [{"shot_id": "shot-0", "source_frame": 0}, {"shot_id": "shot-1", "source_frame": 10}]}), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError, match="source sha256"):
        load_play_alignment(path, source_sha256="actual")


def test_alignment_rejects_unsafe_play_id() -> None:
    with pytest.raises(ReplayAlignmentError, match="safe identifier"):
        PlayAlignment("../escape", (PlayAnchor("shot-0", 0, "snap"), PlayAnchor("shot-1", 10, "snap")))


def test_alignment_loader_marks_matching_source_hash_validated(tmp_path) -> None:
    path = tmp_path / "good-source.json"
    path.write_text(json.dumps({"reviewed": True, "source_sha256": "expected", "play_id": "p1", "anchors": [{"shot_id": "shot-0", "source_frame": 0}, {"shot_id": "shot-1", "source_frame": 10}]}), encoding="utf-8")
    loaded = load_play_alignment(path, source_sha256="expected")
    assert loaded.source_hash_validated is True
    assert loaded.source_sha256 == "expected"


def test_alignment_loader_rejects_anchor_outside_declared_shot_range(tmp_path) -> None:
    path = tmp_path / "outside-range.json"
    path.write_text(json.dumps({"reviewed": True, "play_id": "p1", "anchors": [{"shot_id": "shot-0", "source_frame": 10}, {"shot_id": "shot-1", "source_frame": 20}], "shot_ranges": {"shot-0": [0, 10], "shot-1": [20, 30]}}), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError, match="outside its declared shot range"):
        load_play_alignment(path)


from football_tracking.identity import TeamEvidence
from football_tracking.replay import FieldTrack, cross_shot_candidate_evidence, cross_shot_candidate_scores


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


def test_high_position_uncertainty_abstains_in_candidate_generation() -> None:
    left = [FieldTrack("shot-0:t1", "shot-0", tuple((index * 0.1, 10.0, 20.0) for index in range(8)), (3.0,) * 8)]
    right = [FieldTrack("shot-1:t1", "shot-1", tuple((index * 0.1, 10.0, 20.0) for index in range(8)), (3.0,) * 8)]
    evidence = teams(**{"shot-0:t1": "DET", "shot-1:t1": "DET"})
    assert cross_shot_candidate_scores(left, right, evidence) == {}


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


def test_sample_pairing_keeps_right_times_monotonic() -> None:
    left = FieldTrack("shot-0:t1", "shot-0", ((0.021, 0.0, 0.0), (0.022, 1.0, 0.0)))
    right = FieldTrack("shot-1:t1", "shot-1", ((0.0, 0.0, 0.0), (0.03, 1.0, 0.0)))
    from football_tracking import replay

    assert replay._paired_indices(left, right, 0.05) == [(0, 1)]


def test_candidate_evidence_retains_rejection_reason_and_overlap_metrics() -> None:
    left = [line("shot-0:t1", "shot-0", 10.0, 20.0, count=3)]
    right = [line("shot-1:t1", "shot-1", 10.0, 20.0, count=3)]
    evidence = teams(**{"shot-0:t1": "DET", "shot-1:t1": "DET"})
    report = cross_shot_candidate_evidence(left, right, evidence)
    record = report[("shot-0:t1", "shot-1:t1")]
    assert record["eligible"] is False
    assert record["rejection_reason"] == "insufficient_overlap_samples"
    assert record["paired_sample_count"] == 3
