from __future__ import annotations

import pytest

from football_tracking.calibration import FieldPoint, Homography, ImagePoint, load_calibrations, project_observation
from football_tracking.schema import Observation


def test_homography_round_trips_known_mapping() -> None:
    image = [ImagePoint(0, 0), ImagePoint(100, 0), ImagePoint(100, 100), ImagePoint(0, 100), ImagePoint(50, 50)]
    field = [FieldPoint(0, 0), FieldPoint(10, 0), FieldPoint(10, 10), FieldPoint(0, 10), FieldPoint(5, 5)]

    transform = Homography.fit(image, field)
    projected = transform.project(ImagePoint(25, 75))

    assert projected is not None
    assert projected.x_yards == pytest.approx(2.5, abs=0.01)
    assert projected.y_yards == pytest.approx(7.5, abs=0.01)
    assert transform.median_error_px < 0.01


def test_homography_rejects_degenerate_landmarks() -> None:
    points = [ImagePoint(0, 0), ImagePoint(1, 0), ImagePoint(2, 0), ImagePoint(3, 0)]
    fields = [FieldPoint(0, 0), FieldPoint(1, 0), FieldPoint(2, 0), FieldPoint(3, 0)]
    with pytest.raises(ValueError, match="non-collinear"):
        Homography.fit(points, fields)


def test_project_observation_uses_bottom_center_and_keeps_metadata() -> None:
    transform = Homography.fit(
        [ImagePoint(0, 0), ImagePoint(100, 0), ImagePoint(100, 100), ImagePoint(0, 100)],
        [FieldPoint(0, 0), FieldPoint(10, 0), FieldPoint(10, 10), FieldPoint(0, 10)],
    )
    observation = Observation(
        run_id="r",
        shot_id="s",
        frame_index=1,
        pts=1,
        time_base=(1, 30),
        tracklet_id="s:t1",
        player_id="P01",
        bbox_xyxy_px=(20, 20, 40, 80),
        detection_score=0.9,
        team="DET",
        team_score=0.9,
        jersey_number=None,
        field_xy_yards=None,
        position_source=None,
        calibration_id=None,
        identity_version=1,
    )

    projected = project_observation(observation, transform, source="bottom_center")

    assert projected.field_xy_yards == pytest.approx((3.0, 8.0), abs=0.01)
    assert projected.position_source == "bottom_center"
    assert projected.calibration_id == transform.calibration_id
    assert projected.player_id == "P01"


def test_load_calibrations_supports_shot_specific_landmarks(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        '{"shots":{"shot-0":{"image_points":[[0,0],[100,0],[100,100],[0,100]],"field_points":[[0,0],[10,0],[10,10],[0,10]]},"shot-1":{"image_points":[[0,0],[200,0],[200,200],[0,200]],"field_points":[[0,0],[20,0],[20,20],[0,20]]}}}'
    )

    calibrations = load_calibrations(path)

    assert set(calibrations) == {"shot-0", "shot-1"}
    assert calibrations["shot-1"].project(ImagePoint(100, 100)) == FieldPoint(10.0, 10.0)
