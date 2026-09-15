from __future__ import annotations

import json

import pytest

from football_tracking.evaluation import EvaluationError, evaluate_tracking, load_reviewed_mot_reference, load_cross_shot_identity, evaluate_cross_shot_identity
from football_tracking.tracking import TrackObservation


def track(tracklet_id: str, frame_index: int, box=(0.0, 0.0, 10.0, 10.0)) -> TrackObservation:
    return TrackObservation(tracklet_id, frame_index, frame_index * 1001, box, 0.9)


def test_reviewed_reference_uses_one_based_frame_mapping_and_ignored_frames(tmp_path) -> None:
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps({
        "reviewed": True,
        "sequences": {"shot-0": {"frames": {
            "10": {"labeled": True, "objects": [{"id": "p1", "bbox_xyxy": [0, 0, 10, 10]}]},
            "11": {"labeled": False, "objects": []},
            "12": {"labeled": True, "ignore": True, "objects": [{"id": "p1", "bbox_xyxy": [0, 0, 10, 10]}]},
        }}},
    }), encoding="utf-8")

    reference = load_reviewed_mot_reference(reference_path)
    report = evaluate_tracking([track("shot-0:t1", 10), track("shot-0:t1", 11), track("shot-0:t1", 12)], reference)

    assert reference["shot-0"][10].evaluator_frame == 11
    assert report["aggregate"]["coverage"] == pytest.approx(1.0)
    assert report["aggregate"]["evaluated_frames"] == 1


def test_evaluator_distinguishes_perfect_swap_and_gap_sequences(tmp_path) -> None:
    reference_path = tmp_path / "reference.json"
    frame = {"labeled": True, "objects": [
        {"id": "a", "bbox_xyxy": [0, 0, 10, 10]},
        {"id": "b", "bbox_xyxy": [20, 0, 30, 10]},
    ]}
    reference_path.write_text(json.dumps({"reviewed": True, "sequences": {"shot-0": {"frames": {
        "0": frame, "1": frame, "2": frame,
    }}}}), encoding="utf-8")
    reference = load_reviewed_mot_reference(reference_path)

    perfect = evaluate_tracking([track("shot-0:t1", index) for index in range(3)] + [track("shot-0:t2", index, (20, 0, 30, 10)) for index in range(3)], reference)
    swapped = evaluate_tracking([track("shot-0:t1", 0), track("shot-0:t2", 0, (20, 0, 30, 10)), track("shot-0:t2", 1), track("shot-0:t1", 1, (20, 0, 30, 10)), track("shot-0:t1", 2), track("shot-0:t2", 2, (20, 0, 30, 10))], reference)
    gapped = evaluate_tracking([track("shot-0:t1", 0), track("shot-0:t1", 2), track("shot-0:t2", 0, (20, 0, 30, 10)), track("shot-0:t2", 1, (20, 0, 30, 10)), track("shot-0:t2", 2, (20, 0, 30, 10))], reference)

    assert perfect["aggregate"]["id_switches"] == 0
    assert swapped["aggregate"]["id_switches"] == 4
    assert gapped["aggregate"]["fragmentation"] == 1


def test_reference_rejects_unreviewed_data(tmp_path) -> None:
    path = tmp_path / "unreviewed.json"
    path.write_text('{"reviewed": false, "sequences": {}}', encoding="utf-8")

    with pytest.raises(EvaluationError, match="reviewed"):
        load_reviewed_mot_reference(path)


def test_evaluator_emits_trackeval_metrics_when_optional_dependency_is_installed(tmp_path) -> None:
    pytest.importorskip("trackeval")
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"reviewed": True, "sequences": {"shot-0": {"frames": {
        "0": {"labeled": True, "objects": [{"id": "a", "bbox_xyxy": [0, 0, 10, 10]}]},
    }}}}), encoding="utf-8")

    report = evaluate_tracking([track("shot-0:t1", 0)], load_reviewed_mot_reference(path))

    assert report["per_shot"]["shot-0"]["trackeval"]["status"] == "evaluated"
    assert report["per_shot"]["shot-0"]["trackeval"]["IDF1"] == pytest.approx(1.0)


def test_cross_shot_identity_loader_requires_reviewed_and_returns_none_when_absent(tmp_path) -> None:
    without = tmp_path / "without.json"
    without.write_text(json.dumps({"reviewed": True, "sequences": {"shot-0": {"frames": {}}}}), encoding="utf-8")
    assert load_cross_shot_identity(without) is None

    with_map = tmp_path / "with.json"
    with_map.write_text(json.dumps({
        "reviewed": True,
        "sequences": {"shot-0": {"frames": {}}},
        "cross_shot_identity": {"shot-0": {"p1": "PLAYER-A"}, "shot-1": {"p9": "PLAYER-A"}},
    }), encoding="utf-8")
    assert load_cross_shot_identity(with_map) == {"shot-0": {"p1": "PLAYER-A"}, "shot-1": {"p9": "PLAYER-A"}}

    unreviewed = tmp_path / "unreviewed.json"
    unreviewed.write_text(json.dumps({"cross_shot_identity": {"shot-0": {"p1": "PLAYER-A"}}}), encoding="utf-8")
    with pytest.raises(EvaluationError):
        load_cross_shot_identity(unreviewed)


def _two_shot_reference(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({
        "reviewed": True,
        "sequences": {
            "shot-0": {"frames": {
                "0": {"labeled": True, "objects": [
                    {"id": "p1", "bbox_xyxy": [0, 0, 10, 10]},
                    {"id": "p2", "bbox_xyxy": [50, 50, 60, 60]},
                ]},
            }},
            "shot-1": {"frames": {
                "100": {"labeled": True, "objects": [
                    {"id": "q1", "bbox_xyxy": [0, 0, 10, 10]},
                    {"id": "q2", "bbox_xyxy": [50, 50, 60, 60]},
                ]},
            }},
        },
        "cross_shot_identity": {
            "shot-0": {"p1": "PLAYER-A", "p2": "PLAYER-B"},
            "shot-1": {"q1": "PLAYER-A", "q2": "PLAYER-B"},
        },
    }), encoding="utf-8")
    return path


def test_cross_shot_evaluation_reports_zero_coverage_for_unlinked_identities(tmp_path) -> None:
    path = _two_shot_reference(tmp_path)
    reference = load_reviewed_mot_reference(path)
    links = load_cross_shot_identity(path)
    predictions = [
        track("shot-0:t1", 0, (0.0, 0.0, 10.0, 10.0)),
        track("shot-0:t2", 0, (50.0, 50.0, 60.0, 60.0)),
        track("shot-1:t1", 100, (0.0, 0.0, 10.0, 10.0)),
        track("shot-1:t2", 100, (50.0, 50.0, 60.0, 60.0)),
    ]
    identity_map = {"shot-0:t1": "P01", "shot-0:t2": "P02", "shot-1:t1": "P03", "shot-1:t2": "P04"}

    report = evaluate_cross_shot_identity(identity_map, predictions, reference, links)

    assert report["status"] == "evaluated"
    assert report["resolvable_pairs"] == 2
    assert report["merged_pairs"] == 0
    assert report["true_merges"] == 0
    assert report["false_merges"] == 0
    assert report["coverage"] == pytest.approx(0.0)
    assert report["precision"] is None


def test_cross_shot_evaluation_separates_true_and_false_merges(tmp_path) -> None:
    path = _two_shot_reference(tmp_path)
    reference = load_reviewed_mot_reference(path)
    links = load_cross_shot_identity(path)
    predictions = [
        track("shot-0:t1", 0, (0.0, 0.0, 10.0, 10.0)),
        track("shot-0:t2", 0, (50.0, 50.0, 60.0, 60.0)),
        track("shot-1:t1", 100, (0.0, 0.0, 10.0, 10.0)),
        track("shot-1:t2", 100, (50.0, 50.0, 60.0, 60.0)),
    ]
    # t1 pair is correct (both PLAYER-A); t2 of shot-0 is wrongly merged with t1 of shot-1.
    identity_map = {"shot-0:t1": "P01", "shot-1:t1": "P01", "shot-0:t2": "P02", "shot-1:t2": "P02"}
    good = evaluate_cross_shot_identity(identity_map, predictions, reference, links)
    assert good["true_merges"] == 2
    assert good["false_merges"] == 0
    assert good["coverage"] == pytest.approx(1.0)
    assert good["precision"] == pytest.approx(1.0)

    wrong = {"shot-0:t2": "P01", "shot-1:t1": "P01", "shot-0:t1": "P02", "shot-1:t2": "P02"}
    bad = evaluate_cross_shot_identity(wrong, predictions, reference, links)
    assert bad["false_merges"] == 2
    assert bad["true_merges"] == 0
    assert bad["precision"] == pytest.approx(0.0)


def test_cross_shot_evaluation_is_not_evaluated_without_identity_map(tmp_path) -> None:
    path = _two_shot_reference(tmp_path)
    reference = load_reviewed_mot_reference(path)
    report = evaluate_cross_shot_identity({}, [], reference, None)
    assert report["status"] == "not_evaluated"
    assert "cross_shot_identity" in report["reason"]


def test_cross_shot_coverage_denominator_keeps_reviewed_player_missed_by_detector(tmp_path) -> None:
    path = _two_shot_reference(tmp_path)
    reference = load_reviewed_mot_reference(path)
    links = load_cross_shot_identity(path)
    predictions = [
        track("shot-0:t1", 0, (0.0, 0.0, 10.0, 10.0)),
        track("shot-1:t1", 100, (0.0, 0.0, 10.0, 10.0)),
    ]
    report = evaluate_cross_shot_identity({"shot-0:t1": "P01", "shot-1:t1": "P01"}, predictions, reference, links)
    assert report["shared_player_coverage"] == {"correct": 1, "eligible": 2, "value": 0.5}
    assert report["gate"]["coverage_at_least_0_80"] is False
    assert report["missing_shared_players"] == ["PLAYER-B"]
