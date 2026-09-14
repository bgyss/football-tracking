from __future__ import annotations

import json

import pytest

from football_tracking.evaluation import EvaluationError, evaluate_tracking, load_reviewed_mot_reference
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
