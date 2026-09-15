from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from football_tracking.calibration_timeline import CalibrationTimelineError, estimate_field_motion, load_calibration_timeline, propagate_calibration_timeline, propagate_homography, static_field_mask, timeline_from_landmark_records
from football_tracking.calibration import FieldPoint, Homography, ImagePoint, project_observation
from football_tracking.schema import Observation


def config(scale: float = 10.0) -> dict:
    return {"image_points": [[0, 0], [100, 0], [100, 100], [0, 100]], "field_points": [[0, 0], [scale, 0], [scale, scale], [0, scale]], "withheld_image_points": [[50, 50]], "withheld_field_points": [[scale / 2, scale / 2]]}


def test_timeline_is_pts_scoped_and_rejects_gaps(tmp_path) -> None:
    path = tmp_path / "timeline.json"
    first = {**config(), "keyframe_id": "early", "pts_start": 0, "pts_end": 100}
    second = {**config(), "keyframe_id": "late", "pts_start": 100, "pts_end": 200}
    path.write_text(json.dumps({"schema_version": 2, "shots": {"shot-0": {"keyframes": [first, second]}}}), encoding="utf-8")
    timeline = load_calibration_timeline(path)
    assert timeline.at("shot-0", 50).calibration_id != timeline.at("shot-0", 150).calibration_id
    assert timeline.at("shot-0", 250) is None
    assert timeline.at("shot-1", 50) is None


def test_timeline_derives_end_of_open_keyframe_from_next_keyframe(tmp_path) -> None:
    path = tmp_path / "timeline.json"
    first = {**config(), "keyframe_id": "early", "pts_start": 0}
    second = {**config(), "keyframe_id": "late", "pts_start": 100, "pts_end": 200}
    path.write_text(json.dumps({"schema_version": 2, "shots": {"shot-0": {"keyframes": [first, second]}}}), encoding="utf-8")
    timeline = load_calibration_timeline(path)
    assert timeline.at("shot-0", 99).calibration_id != timeline.at("shot-0", 101).calibration_id


def test_timeline_export_round_trips_validity(tmp_path) -> None:
    path = tmp_path / "timeline.json"
    value = {"schema_version": 2, "shots": {"shot-0": {"keyframes": [{**config(), "pts_start": 0, "pts_end": 100}]}}}
    path.write_text(json.dumps(value), encoding="utf-8")
    timeline = load_calibration_timeline(path)
    exported = tmp_path / "exported.json"
    exported.write_text(json.dumps(timeline.to_dict()), encoding="utf-8")
    reloaded = load_calibration_timeline(exported)
    assert reloaded.at("shot-0", 50).identity_eligible is True


def test_timeline_marks_unvalidated_intervals_ineligible_even_with_valid_neighbor(tmp_path) -> None:
    path = tmp_path / "mixed.json"
    valid = {**config(), "pts_start": 0, "pts_end": 100}
    unvalidated = {"image_points": [[0, 0], [100, 0], [100, 100], [0, 100]], "field_points": [[0, 0], [10, 0], [10, 10], [0, 10]], "pts_start": 100, "pts_end": 200}
    path.write_text(json.dumps({"schema_version": 2, "shots": {"shot-0": {"keyframes": [valid, unvalidated]}}}), encoding="utf-8")
    timeline = load_calibration_timeline(path)
    assert timeline.at("shot-0", 50).identity_eligible is True
    assert timeline.at("shot-0", 150).identity_eligible is False


def test_support_polygon_invalidates_contact_outside_camera_support() -> None:
    h = Homography.fit([ImagePoint(0, 0), ImagePoint(100, 0), ImagePoint(100, 100), ImagePoint(0, 100)], [FieldPoint(0, 0), FieldPoint(10, 0), FieldPoint(10, 10), FieldPoint(0, 10)])
    observation = Observation(run_id="r", shot_id="s", frame_index=0, pts=0, time_base=(1, 60), tracklet_id="s:t", player_id=None, bbox_xyxy_px=(90, 90, 95, 95), detection_score=0.9, team="unknown", team_score=0.0, jersey_number=None, field_xy_yards=None, position_source=None, calibration_id=None, identity_version=1)
    projected = project_observation(observation, h, support_polygon_px=((0, 0), (80, 0), (80, 80), (0, 80)))
    assert projected.field_xy_yards is None
    assert projected.calibration_status == "invalid"


def test_timeline_from_reviewed_landmarks_requires_withheld_points() -> None:
    records = []
    image = [(0, 0), (100, 0), (100, 100), (0, 100)]
    field = [(0, 0), (10, 0), (10, 10), (0, 10)]
    for index, (image_point, field_point) in enumerate(zip(image, field)):
        records.append({"shot_id": "shot-0", "source_frame": 10, "source_pts": 1000, "image_xy_px": image_point, "field_xy_yards": field_point, "role": "fit", "id": f"fit-{index}"})
    records.append({"shot_id": "shot-0", "source_frame": 10, "source_pts": 1000, "image_xy_px": (50, 50), "field_xy_yards": (5, 5), "role": "withheld", "id": "withheld"})
    timeline = timeline_from_landmark_records(records)
    assert timeline.at("shot-0", 1000).identity_eligible is True
    with pytest.raises(CalibrationTimelineError, match="withheld"):
        timeline_from_landmark_records([record for record in records if record["role"] == "fit"])


def test_timeline_rejects_mirrored_semantic_landmark_coordinates() -> None:
    records = []
    image = [(0, 0), (100, 0), (100, 100), (0, 100)]
    field = [(0, 0), (10, 0), (10, 10), (0, 10)]
    ids = ["yardline:20:sideline:near", "yardline:40:sideline:near", "yardline:40:sideline:far", "yardline:20:sideline:far"]
    for index, (image_point, field_point) in enumerate(zip(image, field)):
        records.append({"id": f"fit-{index}", "landmark_id": ids[index], "shot_id": "shot-0", "source_frame": 10, "source_pts": 1000, "image_xy_px": image_point, "field_xy_yards": field_point, "role": "fit"})
    records.append({"id": "withheld", "landmark_id": "yardline:30:sideline:near", "shot_id": "shot-0", "source_frame": 10, "source_pts": 1000, "image_xy_px": (50, 50), "field_xy_yards": (30, 0), "role": "withheld"})
    with pytest.raises(CalibrationTimelineError, match="semantic landmark"):
        timeline_from_landmark_records(records)


def test_timeline_rejects_overlapping_intervals(tmp_path) -> None:
    path = tmp_path / "timeline.json"
    first = {**config(), "pts_start": 0, "pts_end": 100}
    second = {**config(), "pts_start": 50, "pts_end": 200}
    path.write_text(json.dumps({"schema_version": 2, "shots": {"shot-0": {"keyframes": [first, second]}}}), encoding="utf-8")
    with pytest.raises(CalibrationTimelineError, match="overlapping"):
        load_calibration_timeline(path)


def test_timeline_rejects_declared_source_hash_mismatch(tmp_path) -> None:
    path = tmp_path / "timeline.json"
    path.write_text(json.dumps({"schema_version": 2, "source_sha256": "expected", "shots": {"shot-0": {"keyframes": [{**config(), "pts_start": 0, "pts_end": 100}]}}}), encoding="utf-8")
    with pytest.raises(CalibrationTimelineError, match="sha256"):
        load_calibration_timeline(path, source_sha256="actual")


def test_propagation_composes_current_to_keyframe_direction() -> None:
    h = Homography.fit([ImagePoint(0, 0), ImagePoint(100, 0), ImagePoint(100, 100), ImagePoint(0, 100)], [FieldPoint(0, 0), FieldPoint(10, 0), FieldPoint(10, 10), FieldPoint(0, 10)])
    # Current pixels are keyframe pixels translated by +10, so current->field
    # must first translate by -10 before applying the keyframe H.
    motion = np.asarray([[1, 0, 10], [0, 1, 0], [0, 0, 1]], dtype=float)
    propagated = propagate_homography(h, motion)
    assert propagated.project(ImagePoint(10, 0)) == FieldPoint(0.0, 0.0)


def test_timeline_motion_propagation_stops_after_a_large_gap() -> None:
    h = Homography.fit(
        [ImagePoint(0, 0), ImagePoint(100, 0), ImagePoint(100, 100), ImagePoint(0, 100)],
        [FieldPoint(0, 0), FieldPoint(10, 0), FieldPoint(10, 10), FieldPoint(0, 10)],
        withheld_image_points=[ImagePoint(50, 50)],
        withheld_field_points=[FieldPoint(5, 5)],
    )
    from football_tracking.calibration_timeline import CalibrationEstimate, CalibrationTimeline

    estimate = CalibrationEstimate("shot-0", "key", h, 0, None, ("key",), "valid", support_polygon_px=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)))
    timeline = CalibrationTimeline({"shot-0": [estimate]})
    propagated = propagate_calibration_timeline(
        timeline,
        {"shot-0": [(10, ((1, 0, 1), (0, 1, 0), (0, 0, 1))), (30, ((1, 0, 1), (0, 1, 0), (0, 0, 1)))]},
        max_gap_pts=12,
    )
    assert propagated.at("shot-0", 5) is not None
    assert propagated.at("shot-0", 10) is not None
    assert propagated.at("shot-0", 21) is not None
    assert propagated.at("shot-0", 22) is None
    assert propagated.at("shot-0", 30) is None
    assert propagated.at("shot-0", 10).support_polygon_px[0] == (1.0, 0.0)


def test_estimate_field_motion_returns_none_without_spatial_support() -> None:
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    assert estimate_field_motion(frame, frame) is None


def test_static_field_mask_excludes_dynamic_boxes_with_margin() -> None:
    mask = static_field_mask((20, 30), [(10, 5, 15, 10)], margin_px=2)
    assert mask[5, 10] == 0
    assert mask[0, 0] == 255


def test_estimate_field_motion_recovers_a_translation_from_static_texture() -> None:
    previous = np.zeros((120, 160, 3), dtype=np.uint8)
    for y in range(10, 110, 15):
        for x in range(10, 150, 15):
            cv2.circle(previous, (x, y), 3, (255, 255, 255), -1)
    current = np.roll(previous, 5, axis=1)
    motion = estimate_field_motion(previous, current)
    assert motion is not None
    assert motion[0, 2] == pytest.approx(5.0, abs=1.5)
