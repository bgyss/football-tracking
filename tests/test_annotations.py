from __future__ import annotations

import json

import pytest

from football_tracking.annotations import AnnotationError, load_annotation_manifest, source_bbox_from_crop


def source() -> dict:
    return {"sha256": "abc", "width": 1920, "height": 1080, "frame_count": 100, "time_base": [1, 60]}


def manifest(**updates) -> dict:
    value = {
        "schema_version": 1,
        "reviewed": True,
        "source": source(),
        "shots": {"shot-0": {"start_frame": 0, "end_frame": 10, "play_id": "p1", "split": "development"}},
        "annotations": [{"id": "a", "shot_id": "shot-0", "source_frame": 2, "bbox_xyxy_px": [1, 2, 10, 20], "review_status": "reviewed", "coordinate_space": "source"}],
    }
    value.update(updates)
    return value


def write(tmp_path, value):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_load_annotation_manifest_validates_source_and_review_state(tmp_path) -> None:
    parsed = load_annotation_manifest(write(tmp_path, manifest()), "abc")
    assert parsed.shots["shot-0"].contains(2)
    with pytest.raises(AnnotationError, match="sha256"):
        load_annotation_manifest(write(tmp_path, manifest()), "wrong")
    unreviewed = manifest(reviewed=False)
    with pytest.raises(AnnotationError, match="reviewed"):
        load_annotation_manifest(write(tmp_path, unreviewed), "abc")


def test_load_annotation_manifest_rejects_duplicates_bounds_and_split_conflicts(tmp_path) -> None:
    duplicate = manifest(annotations=[manifest()["annotations"][0], manifest()["annotations"][0]])
    with pytest.raises(AnnotationError, match="duplicate"):
        load_annotation_manifest(write(tmp_path, duplicate), "abc")
    out_of_shot = manifest(annotations=[{**manifest()["annotations"][0], "source_frame": 20}])
    with pytest.raises(AnnotationError, match="outside"):
        load_annotation_manifest(write(tmp_path, out_of_shot), "abc")
    conflict = manifest(shots={"shot-0": {"start_frame": 0, "end_frame": 10, "play_id": "p1", "split": "development"}, "shot-1": {"start_frame": 10, "end_frame": 20, "play_id": "p1", "split": "test"}})
    with pytest.raises(AnnotationError, match="conflicting"):
        load_annotation_manifest(write(tmp_path, conflict), "abc")


def test_source_bbox_from_crop_round_trips_resized_coordinates() -> None:
    assert source_bbox_from_crop((10, 20, 30, 60), (100, 200, 500, 600), (200, 200)) == (120.0, 240.0, 160.0, 320.0)


def test_load_annotation_manifest_validates_fit_and_withheld_landmarks(tmp_path) -> None:
    value = manifest(landmarks=[{"id": "yardline-20-near", "shot_id": "shot-0", "source_frame": 2, "image_xy_px": [100, 200], "field_xy_yards": [20, 0], "role": "fit", "review_status": "reviewed"}])
    parsed = load_annotation_manifest(write(tmp_path, value), "abc")
    assert parsed.landmarks[0]["role"] == "fit"
    invalid = manifest(landmarks=[{"id": "bad", "shot_id": "shot-0", "source_frame": 2, "image_xy_px": [100, 200], "field_xy_yards": [20, 0], "role": "fit", "review_status": "unreviewed"}])
    with pytest.raises(AnnotationError, match="not reviewed"):
        load_annotation_manifest(write(tmp_path, invalid), "abc")
