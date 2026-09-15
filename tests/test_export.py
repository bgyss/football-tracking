from __future__ import annotations

import csv
import json

import cv2
import numpy as np

from football_tracking.export import (
    StageTimer,
    render_annotated_video,
    write_field_view,
    write_identities_json,
    write_observations_parquet,
    write_observations_csv,
    write_trajectories_csv,
)
from football_tracking.schema import Observation
from football_tracking.video import ShotBoundary


def observation(frame: int, shot: str, player: str = "P01") -> Observation:
    return Observation(
        run_id="run",
        shot_id=shot,
        frame_index=frame,
        pts=frame,
        time_base=(1, 10),
        tracklet_id=f"{shot}:t1",
        player_id=player,
        bbox_xyxy_px=(2.0 + frame, 2.0, 8.0 + frame, 12.0),
        detection_score=0.9,
        team="DET",
        team_score=0.9,
        jersey_number=None,
        field_xy_yards=(float(frame), 2.0),
        position_source="bottom_center",
        calibration_id="cal",
        identity_version=1,
    )


def test_table_and_json_exports_are_deterministic(tmp_path) -> None:
    rows = [observation(1, "shot-1"), observation(0, "shot-0")]
    csv_path = tmp_path / "observations.csv"
    identities_path = tmp_path / "identities.json"

    write_observations_csv(csv_path, rows)
    write_identities_json(identities_path, {"shot-1:t1": "P01"})

    with csv_path.open(newline="") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed[0]["frame_index"] == "0"
    assert json.loads(parsed[0]["field_xy_yards"]) == [0.0, 2.0]
    assert json.loads(identities_path.read_text()) == {"shot-1:t1": "P01"}


def test_annotated_video_resets_trails_at_shot_boundary(tmp_path) -> None:
    input_path = tmp_path / "tiny.mp4"
    output_path = tmp_path / "annotated.mp4"
    writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()

    render_annotated_video(input_path, output_path, [observation(0, "shot-0"), observation(2, "shot-1")], [ShotBoundary(0, 0, "start", 1.0), ShotBoundary(2, 2, "manual", 1.0)])

    capture = cv2.VideoCapture(str(output_path))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    assert frames == 4


def test_field_view_and_stage_timer(tmp_path) -> None:
    output = tmp_path / "field.png"
    write_field_view(output, {("P01", "shot-0"): [(0.0, 1.0), (10.0, 5.0)]})
    timer = StageTimer()
    with timer.stage("unit"):
        pass
    assert output.stat().st_size > 0
    assert timer.timings["unit_s"] >= 0.0


def test_field_view_draws_a_replay_merged_player_as_two_separate_paths(tmp_path) -> None:
    # A player observed in both shots of a correct replay merge must be drawn as two
    # independent trails, not one line joined across the cut by a spurious segment.
    output_split = tmp_path / "field-split.png"
    output_joined = tmp_path / "field-joined.png"

    write_field_view(output_split, {
        ("P01", "shot-0"): [(0.0, 1.0), (10.0, 1.0)],
        ("P01", "shot-1"): [(80.0, 40.0), (90.0, 40.0)],
    })
    # The old behavior: one continuous path across both shots' points, in tracklet
    # insertion order, would draw a segment bridging (10, 1) -> (80, 40).
    write_field_view(output_joined, {("P01", "shot-0"): [(0.0, 1.0), (10.0, 1.0), (80.0, 40.0), (90.0, 40.0)]})

    split = cv2.imread(str(output_split))
    joined = cv2.imread(str(output_joined))
    assert split is not None and joined is not None
    # A pixel near the midpoint of the spurious bridging segment is painted in the
    # joined rendering but must stay background-colored in the split rendering.
    midpoint_yards = (45.0, 20.5)
    margin_x, margin_y = 70, 60
    field_width, field_height = 1200 - margin_x * 2, 560 - margin_y * 2
    px = margin_x + int(field_width * midpoint_yards[0] / 120.0)
    py = margin_y + field_height - int(field_height * midpoint_yards[1] / (160.0 / 3.0))
    background = tuple(int(value) for value in (35, 105, 35))
    assert tuple(int(value) for value in split[py, px]) == background
    assert tuple(int(value) for value in joined[py, px]) != background


def test_parquet_export_contains_observation_contract(tmp_path) -> None:
    path = tmp_path / "observations.parquet"

    write_observations_parquet(path, [observation(0, "shot-0")])

    import pyarrow.parquet as parquet

    table = parquet.read_table(path)
    assert "frame_index" in table.column_names
    assert table.column("player_id").to_pylist() == ["P01"]


def test_trajectory_export_contains_pixel_positions_and_optional_field_positions(tmp_path) -> None:
    path = tmp_path / "trajectories.csv"
    invalid = observation(1, "shot-0")
    invalid = invalid.__class__(**{**invalid.to_dict(), "field_xy_yards": None})

    write_trajectories_csv(path, [invalid, observation(0, "shot-0")])

    with path.open(newline="") as handle:
        parsed = list(csv.DictReader(handle))
    assert len(parsed) == 2
    assert parsed[0]["player_id"] == "P01"
    assert parsed[0]["x_px"] == "5.0"
    assert parsed[0]["x_yards"] == "0.0"
    assert parsed[1]["x_yards"] == ""
