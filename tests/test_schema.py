from __future__ import annotations

import json

import pytest

from football_tracking.schema import Observation, RunManifest


def test_observation_serializes_nullable_values_and_coordinates() -> None:
    observation = Observation(
        run_id="run-1",
        shot_id="shot-0",
        frame_index=4,
        pts=4004,
        time_base=(1, 60000),
        tracklet_id="shot-0:t3",
        player_id=None,
        bbox_xyxy_px=(10.0, 20.0, 30.0, 80.0),
        detection_score=0.91,
        team="DET",
        team_score=0.98,
        jersey_number=None,
        field_xy_yards=None,
        position_source=None,
        calibration_id=None,
        identity_version=1,
    )

    value = observation.to_dict()
    assert value["bbox_xyxy_px"] == [10.0, 20.0, 30.0, 80.0]
    assert value["player_id"] is None
    assert value["time_base"] == [1, 60000]
    assert json.loads(json.dumps(value)) == value


def test_observation_rejects_invalid_box() -> None:
    with pytest.raises(ValueError, match="bbox"):
        Observation(
            run_id="run-1",
            shot_id="shot-0",
            frame_index=4,
            pts=4004,
            time_base=(1, 60000),
            tracklet_id="shot-0:t3",
            player_id=None,
            bbox_xyxy_px=(30.0, 20.0, 10.0, 80.0),
            detection_score=0.91,
            team="unknown",
            team_score=0.0,
            jersey_number=None,
            field_xy_yards=None,
            position_source=None,
            calibration_id=None,
            identity_version=1,
        )


def test_manifest_round_trips_with_sorted_json_safe_values() -> None:
    manifest = RunManifest(
        run_id="run-1",
        input_path="data/sample.mp4",
        input_sha256="abc",
        input_duration_s=2.0,
        source_fps=59.94,
        frame_count=120,
        detector="rfdetr-small",
        detector_version="1.10.1",
        tracker="botsort",
        tracker_version="2.6.0",
        config_hash="cfg",
        device="cpu",
        status="complete_with_unresolved",
        timings={"decode_s": 1.2},
        api_usage={"requests": 0},
    )
    value = manifest.to_dict()
    assert list(value) == sorted(value)
    assert value["status"] == "complete_with_unresolved"
    assert json.loads(json.dumps(value))["input_sha256"] == "abc"
