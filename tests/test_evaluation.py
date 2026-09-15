from __future__ import annotations

import json

import pytest

from football_tracking.evaluation import EvaluationError, ReferenceFrame, ReferenceObject, evaluate_ground_contact_positions, evaluate_promotion_gates, evaluate_team_assignment, evaluate_tracking, load_reviewed_mot_reference, load_cross_shot_identity, evaluate_cross_shot_identity, restrict_reference_to_window
from football_tracking.schema import Observation
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


def test_standard_identity_metric_is_not_detection_f1_for_alternating_ids(tmp_path) -> None:
    pytest.importorskip("trackeval")
    reference = {
        "shot-0": {
            0: ReferenceFrame(0, 1, True, False, (ReferenceObject("a", (0, 0, 10, 10)), ReferenceObject("b", (20, 0, 30, 10)))),
            1: ReferenceFrame(1, 2, True, False, (ReferenceObject("a", (0, 0, 10, 10)), ReferenceObject("b", (20, 0, 30, 10)))),
        }
    }
    predictions = [track("shot-0:t1", 0), track("shot-0:t2", 0, (20, 0, 30, 10)), track("shot-0:t2", 1), track("shot-0:t1", 1, (20, 0, 30, 10))]
    report = evaluate_tracking(predictions, reference)
    assert report["per_shot"]["shot-0"]["detection_f1"] == pytest.approx(1.0)
    assert report["standard_metrics"]["aggregate"]["IDF1"] < 1.0


def test_reference_rejects_unreviewed_data(tmp_path) -> None:
    path = tmp_path / "unreviewed.json"
    path.write_text('{"reviewed": false, "sequences": {}}', encoding="utf-8")

    with pytest.raises(EvaluationError, match="reviewed"):
        load_reviewed_mot_reference(path)


def test_reference_preserves_pts_and_rejects_invalid_pts(tmp_path) -> None:
    valid = tmp_path / "valid-pts.json"
    valid.write_text(json.dumps({"reviewed": True, "sequences": {"shot-0": {"frames": {"0": {"pts": 100, "labeled": True, "objects": []}}}}}), encoding="utf-8")
    assert load_reviewed_mot_reference(valid)["shot-0"][0].pts == 100
    invalid = tmp_path / "invalid-pts.json"
    invalid.write_text(json.dumps({"reviewed": True, "sequences": {"shot-0": {"frames": {"0": {"pts": "bad", "labeled": True, "objects": []}}}}}), encoding="utf-8")
    with pytest.raises(EvaluationError, match="pts"):
        load_reviewed_mot_reference(invalid)


def test_reference_window_restriction_preserves_shots_and_frame_scope() -> None:
    reference = {"shot-0": {0: ReferenceFrame(0, 1, True, False, ()), 1: ReferenceFrame(1, 2, True, False, ()), 3: ReferenceFrame(3, 4, True, False, ())}, "shot-1": {2: ReferenceFrame(2, 3, True, False, ())}}
    restricted = restrict_reference_to_window(reference, 1, 3)
    assert set(restricted["shot-0"]) == {1}
    assert set(restricted["shot-1"]) == {2}


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


def test_bounded_reference_window_does_not_count_unseen_shots_in_shared_denominator(tmp_path) -> None:
    path = _two_shot_reference(tmp_path)
    reference = restrict_reference_to_window(load_reviewed_mot_reference(path), 0, 1)
    report = evaluate_cross_shot_identity({}, [], reference, load_cross_shot_identity(path))
    assert report["status"] == "not_evaluated"
    assert "at least two shots" in report["reason"]


def test_cross_shot_evaluation_excludes_unknown_truth_from_shared_denominator() -> None:
    reference = {
        "shot-0": {0: ReferenceFrame(0, 1, True, False, (ReferenceObject("a", (0, 0, 10, 10)),))},
        "shot-1": {0: ReferenceFrame(0, 1, True, False, (ReferenceObject("b", (0, 0, 10, 10)),))},
    }
    report = evaluate_cross_shot_identity(
        {},
        [],
        reference,
        {"shot-0": {"a": "unknown"}, "shot-1": {"b": "unknown"}},
    )
    assert report["status"] == "not_evaluated"


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


def test_nonoverlapping_fragments_do_not_count_as_component_contamination() -> None:
    reference = {
        "shot-0": {
            0: ReferenceFrame(0, 1, True, False, (ReferenceObject("a0", (0, 0, 10, 10)),)),
            1: ReferenceFrame(1, 2, True, False, (ReferenceObject("a1", (0, 0, 10, 10)),)),
        },
        "shot-1": {10: ReferenceFrame(10, 11, True, False, (ReferenceObject("b", (0, 0, 10, 10)),))},
    }
    predictions = [track("shot-0:t1", 0), track("shot-0:t2", 1), track("shot-1:t3", 10)]
    truth = {"shot-0": {"a0": "PLAYER-A", "a1": "PLAYER-A"}, "shot-1": {"b": "PLAYER-A"}}
    report = evaluate_cross_shot_identity({"shot-0:t1": "P01", "shot-0:t2": "P01", "shot-1:t3": "P01"}, predictions, reference, truth)
    assert report["component_contamination"] == {}
    assert report["gate"]["components_clean"] is True


def test_promotion_gates_do_not_promote_missing_or_weak_evidence() -> None:
    missing = evaluate_promotion_gates({"standard_metrics": {"status": "unavailable"}}, None, {"status": "valid"})
    assert missing["status"] == "not_evaluated"
    unvalidated = evaluate_promotion_gates({"standard_metrics": {"status": "evaluated", "aggregate": {"IDF1": 0.95}, "per_shot": {"shot-0": {"IDF1": 0.95, "id_switches": 0}}}}, {"status": "evaluated", "false_merges": 0, "coverage": 0.9, "shared_player_coverage": {"correct": 9, "eligible": 10}, "gate": {"components_clean": True, "false_merges_zero": True}}, {"status": "unvalidated"})
    assert unvalidated["status"] == "not_evaluated"
    failed = evaluate_promotion_gates({"standard_metrics": {"status": "evaluated", "aggregate": {"IDF1": 0.95}, "per_shot": {"shot-0": {"IDF1": 0.95, "id_switches": 0}}}}, {"status": "evaluated", "false_merges": 1, "coverage": 1.0, "shared_player_coverage": {"correct": 10, "eligible": 10}, "gate": {"components_clean": True, "false_merges_zero": False}}, {"status": "valid", "gate": {"median_at_most_1_yard": True, "p95_at_most_2_yards": True}})
    assert failed["status"] == "failed"
    passed = evaluate_promotion_gates({"standard_metrics": {"status": "evaluated", "aggregate": {"IDF1": 0.95}, "per_shot": {"shot-0": {"IDF1": 0.95, "id_switches": 0}}}}, {"status": "evaluated", "false_merges": 0, "coverage": 0.9, "shared_player_coverage": {"correct": 9, "eligible": 10}, "gate": {"components_clean": True, "false_merges_zero": True}}, {"status": "valid", "gate": {"median_at_most_1_yard": True, "p95_at_most_2_yards": True}})
    assert passed["status"] == "passed"
    weak_shot = evaluate_promotion_gates({"standard_metrics": {"status": "evaluated", "aggregate": {"IDF1": 0.95}, "per_shot": {"shot-0": {"IDF1": 0.95, "id_switches": 0}, "shot-1": {"IDF1": 0.5, "id_switches": 0}}}}, {"status": "evaluated", "false_merges": 0, "coverage": 0.9, "shared_player_coverage": {"correct": 9, "eligible": 10}, "gate": {"components_clean": True, "false_merges_zero": True}}, {"status": "valid", "gate": {"median_at_most_1_yard": True, "p95_at_most_2_yards": True}})
    assert weak_shot["status"] == "failed"
    proxy = evaluate_promotion_gates({"standard_metrics": {"status": "evaluated", "aggregate": {"IDF1": 0.95}, "per_shot": {"shot-0": {"IDF1": 0.95, "id_switches": 0}}}}, {"status": "evaluated", "false_merges": 0, "coverage": 0.9, "shared_player_coverage": {"correct": 9, "eligible": 10}, "gate": {"components_clean": True, "false_merges_zero": True}}, {"status": "valid", "gate": {"median_at_most_1_yard": True, "p95_at_most_2_yards": True}}, proxy_detector=True)
    assert proxy["status"] == "not_evaluated"


def test_ground_contact_evaluation_reports_position_error_and_missing_contacts() -> None:
    reference = {"shot-0": {0: ReferenceFrame(0, 1, True, False, (ReferenceObject("a", (0, 0, 10, 10), (1.0, 2.0)), ReferenceObject("b", (20, 0, 30, 10), (20.0, 2.0))))}}
    observation = Observation(run_id="r", shot_id="shot-0", frame_index=0, pts=0, time_base=(1, 60), tracklet_id="shot-0:t1", player_id="P01", bbox_xyxy_px=(0, 0, 10, 10), detection_score=0.9, team="DET", team_score=1.0, jersey_number=None, field_xy_yards=(2.0, 2.0), position_source="bottom_center", calibration_id="cal", identity_version=2)
    report = evaluate_ground_contact_positions([observation], reference)
    assert report["status"] == "evaluated"
    assert report["evaluated_contacts"] == 1
    assert report["invalid_or_missing"] == 1
    assert report["median_error_yards"] == 1.0
    assert report["gate"]["every_shot_gate_passes"] is False


def test_ground_contact_confidence_excludes_uncertain_contacts() -> None:
    reference = {"shot-0": {0: ReferenceFrame(0, 1, True, False, (ReferenceObject("a", (0, 0, 10, 10), (1.0, 2.0), ground_contact_confidence=0.2),))}}
    observation = Observation(run_id="r", shot_id="shot-0", frame_index=0, pts=0, time_base=(1, 60), tracklet_id="shot-0:t1", player_id="P01", bbox_xyxy_px=(0, 0, 10, 10), detection_score=0.9, team="DET", team_score=1.0, jersey_number=None, field_xy_yards=(1.0, 2.0), position_source="bottom_center", calibration_id="cal", identity_version=2)
    report = evaluate_ground_contact_positions([observation], reference)
    assert report["status"] == "not_evaluated"
    assert report["excluded_low_confidence"] == 1


def test_team_assignment_reports_accuracy_and_coverage() -> None:
    reference = {"shot-0": {0: ReferenceFrame(0, 1, True, False, (ReferenceObject("a", (0, 0, 10, 10), team="DET"), ReferenceObject("b", (20, 0, 30, 10), team="LAR")))}}
    observations = [
        Observation(run_id="r", shot_id="shot-0", frame_index=0, pts=0, time_base=(1, 60), tracklet_id="shot-0:t1", player_id="P01", bbox_xyxy_px=(0, 0, 10, 10), detection_score=0.9, team="DET", team_score=1.0, jersey_number=None, field_xy_yards=None, position_source=None, calibration_id=None, identity_version=2),
        Observation(run_id="r", shot_id="shot-0", frame_index=0, pts=0, time_base=(1, 60), tracklet_id="shot-0:t2", player_id="P02", bbox_xyxy_px=(20, 0, 30, 10), detection_score=0.9, team="DET", team_score=1.0, jersey_number=None, field_xy_yards=None, position_source=None, calibration_id=None, identity_version=2),
    ]
    report = evaluate_team_assignment(observations, reference)
    assert report["accuracy"] == 0.5
    assert report["coverage"] == 1.0
    assert report["gate"]["accuracy_at_least_0_98"] is False
    assert report["per_shot"]["shot-0"]["accuracy"] == 0.5
    assert report["gate"]["every_shot_gate_passes"] is False
