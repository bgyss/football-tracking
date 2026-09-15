from __future__ import annotations

import json

import pytest

from football_tracking.calibration import FieldPoint, Homography, ImagePoint, calibration_quality_report, load_calibrations, project_observation
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


def test_calibration_export_round_trips_through_legacy_loader(tmp_path) -> None:
    from football_tracking.export import write_calibration_json

    fitted = Homography.fit(
        [ImagePoint(0, 0), ImagePoint(100, 0), ImagePoint(100, 100), ImagePoint(0, 100)],
        [FieldPoint(0, 0), FieldPoint(10, 0), FieldPoint(10, 10), FieldPoint(0, 10)],
    )
    path = tmp_path / "exported.json"
    write_calibration_json(path, {"shot-0": fitted})
    loaded = load_calibrations(path)
    assert loaded["shot-0"].matrix == fitted.matrix
    assert loaded["shot-0"].status == "unvalidated"


def test_legacy_calibration_rejects_a_declared_source_hash_mismatch(tmp_path) -> None:
    path = tmp_path / "mismatch.json"
    path.write_text(json.dumps({"source_sha256": "expected", "image_points": [[0, 0], [100, 0], [100, 100], [0, 100]], "field_points": [[0, 0], [10, 0], [10, 10], [0, 10]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_calibrations(path, source_sha256="actual")


def test_homography_reports_pixel_and_field_residuals_in_their_own_units() -> None:
    image = [ImagePoint(0, 0), ImagePoint(1000, 0), ImagePoint(1000, 500), ImagePoint(0, 500), ImagePoint(500, 250)]
    field = [FieldPoint(0, 0), FieldPoint(120, 0), FieldPoint(120, 53.333), FieldPoint(0, 53.333), FieldPoint(60, 26.666)]
    transform = Homography.fit(image, field, reprojection_threshold_px=3.0)
    assert transform.fit_error_px == transform.median_error_px
    assert transform.median_error_yards < 0.01
    assert transform.reprojection_threshold_px == 3.0
    assert transform.status == "unvalidated"


def test_homography_uses_withheld_landmarks_for_validation_status() -> None:
    image = [ImagePoint(0, 0), ImagePoint(100, 0), ImagePoint(100, 100), ImagePoint(0, 100)]
    field = [FieldPoint(0, 0), FieldPoint(10, 0), FieldPoint(10, 10), FieldPoint(0, 10)]
    transform = Homography.fit(image, field, withheld_image_points=[ImagePoint(50, 50)], withheld_field_points=[FieldPoint(5, 5)])
    assert transform.status == "valid"
    assert transform.withheld_point_count == 1
    assert transform.withheld_p95_error_yards == pytest.approx(0.0)
    observation = Observation(
        run_id="r", shot_id="s", frame_index=0, pts=0, time_base=(1, 60), tracklet_id="s:t", player_id=None,
        bbox_xyxy_px=(10, 10, 20, 50), detection_score=0.9, team="unknown", team_score=0.0,
        jersey_number=None, field_xy_yards=None, position_source=None, calibration_id=None, identity_version=1,
    )
    projected = project_observation(observation, transform)
    assert projected.position_uncertainty_yards == pytest.approx(0.0)


def test_projection_rejects_nonfinite_or_off_field_contact() -> None:
    transform = Homography.fit(
        [ImagePoint(0, 0), ImagePoint(100, 0), ImagePoint(100, 100), ImagePoint(0, 100)],
        [FieldPoint(0, 0), FieldPoint(10, 0), FieldPoint(10, 10), FieldPoint(0, 10)],
    )
    assert transform.project(ImagePoint(float("nan"), 50)) is None
    assert transform.project(ImagePoint(2000, 50)) is None


def test_calibration_quality_report_requires_withheld_validation() -> None:
    unvalidated = Homography.fit([ImagePoint(0, 0), ImagePoint(100, 0), ImagePoint(100, 100), ImagePoint(0, 100)], [FieldPoint(0, 0), FieldPoint(10, 0), FieldPoint(10, 10), FieldPoint(0, 10)])
    assert calibration_quality_report({"shot-0": unvalidated})["status"] == "unvalidated"
