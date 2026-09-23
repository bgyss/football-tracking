from __future__ import annotations

import json

import pytest

from football_tracking.annotations import AnnotationError, load_annotation_manifest, manifest_template_from_review_pack, mot_reference_from_manifest, source_bbox_from_crop


def source() -> dict:
    return {"sha256": "abc", "width": 1920, "height": 1080, "frame_count": 100, "time_base": [1, 60]}


def manifest(**updates) -> dict:
    value = {
        "schema_version": 1,
        "reviewed": True,
        "source": source(),
        "shots": {"shot-0": {"start_frame": 0, "end_frame": 10, "play_id": "p1", "split": "development", "camera_label": "sideline"}},
        "annotations": [{"id": "a", "shot_id": "shot-0", "source_frame": 2, "pts": 120, "bbox_xyxy_px": [1, 2, 10, 20], "review_status": "reviewed", "coordinate_space": "source", "reviewer": "reviewer-1", "revision": 1, "reviewed_at": "2026-09-15T00:00:00Z", "annotation_confidence": 1.0}],
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
    missing_pts = manifest(annotations=[{key: value for key, value in manifest()["annotations"][0].items() if key != "pts"}])
    with pytest.raises(AnnotationError, match="pts"):
        load_annotation_manifest(write(tmp_path, missing_pts), "abc")
    missing_reviewer = manifest(annotations=[{key: value for key, value in manifest()["annotations"][0].items() if key != "reviewer"}])
    with pytest.raises(AnnotationError, match="reviewer"):
        load_annotation_manifest(write(tmp_path, missing_reviewer), "abc")
    bad_timestamp = manifest(annotations=[{**manifest()["annotations"][0], "reviewed_at": "yesterday"}])
    with pytest.raises(AnnotationError, match="ISO-8601"):
        load_annotation_manifest(write(tmp_path, bad_timestamp), "abc")


def test_load_annotation_manifest_rejects_duplicates_bounds_and_split_conflicts(tmp_path) -> None:
    duplicate = manifest(annotations=[manifest()["annotations"][0], manifest()["annotations"][0]])
    with pytest.raises(AnnotationError, match="duplicate"):
        load_annotation_manifest(write(tmp_path, duplicate), "abc")
    out_of_shot = manifest(annotations=[{**manifest()["annotations"][0], "source_frame": 20}])
    with pytest.raises(AnnotationError, match="outside"):
        load_annotation_manifest(write(tmp_path, out_of_shot), "abc")
    conflict = manifest(shots={"shot-0": {"start_frame": 0, "end_frame": 10, "play_id": "p1", "split": "development", "camera_label": "sideline"}, "shot-1": {"start_frame": 10, "end_frame": 20, "play_id": "p1", "split": "test", "camera_label": "endzone"}})
    with pytest.raises(AnnotationError, match="conflicting"):
        load_annotation_manifest(write(tmp_path, conflict), "abc")


def test_source_bbox_from_crop_round_trips_resized_coordinates() -> None:
    assert source_bbox_from_crop((10, 20, 30, 60), (100, 200, 500, 600), (200, 200)) == (120.0, 240.0, 160.0, 320.0)


def test_load_annotation_manifest_validates_fit_and_withheld_landmarks(tmp_path) -> None:
    value = manifest(landmarks=[{"id": "yardline-20-near", "shot_id": "shot-0", "source_frame": 2, "pts": 120, "image_xy_px": [100, 200], "field_xy_yards": [20, 0], "role": "fit", "review_status": "reviewed", "reviewer": "reviewer-1", "revision": 1, "reviewed_at": "2026-09-15T00:00:00Z", "annotation_confidence": 1.0}])
    parsed = load_annotation_manifest(write(tmp_path, value), "abc")
    assert parsed.landmarks[0]["role"] == "fit"
    invalid = manifest(landmarks=[{"id": "bad", "shot_id": "shot-0", "source_frame": 2, "pts": 120, "image_xy_px": [100, 200], "field_xy_yards": [20, 0], "role": "fit", "review_status": "unreviewed", "reviewer": "reviewer-1", "revision": 1, "reviewed_at": "2026-09-15T00:00:00Z", "annotation_confidence": 1.0}])
    with pytest.raises(AnnotationError, match="not reviewed"):
        load_annotation_manifest(write(tmp_path, invalid), "abc")


def test_manifest_template_preserves_review_pack_source_and_frame_records() -> None:
    pack = {"schema_version": 1, "reviewed": False, "source": source(), "frames": [{"source_frame": 2, "pts": 120}]}
    template = manifest_template_from_review_pack(pack, {"shot-0": {"start_frame": 0, "end_frame": 10, "camera_label": "sideline"}})
    assert template["reviewed"] is False
    assert template["source"]["sha256"] == "abc"
    assert template["review_frames"][0]["pts"] == 120


def test_reviewed_manifest_converts_to_mot_reference_and_cross_shot_map(tmp_path) -> None:
    value = manifest(
        annotations=[
            {**manifest()["annotations"][0], "track_id": "p1", "global_id": "PLAYER-A", "team": "DET", "ground_contact_xy_yards": [20, 2]},
            {**manifest()["annotations"][0], "id": "b", "track_id": "q1", "shot_id": "shot-0", "source_frame": 3, "pts": 180, "global_id": "PLAYER-A", "bbox_xyxy_px": [20, 2, 30, 20], "team": "DET"},
        ],
    )
    parsed = load_annotation_manifest(write(tmp_path, value), "abc")
    reference = mot_reference_from_manifest(parsed)
    assert reference["reviewed"] is True
    assert reference["sequences"]["shot-0"]["frames"]["2"]["objects"][0]["id"] == "p1"
    assert reference["cross_shot_identity"]["shot-0"]["p1"] == "PLAYER-A"


def test_reviewed_frame_labels_preserve_empty_and_ignored_frames(tmp_path) -> None:
    base_annotation = manifest()["annotations"][0]
    frame_meta = {"shot_id": "shot-0", "source_frame": 1, "pts": 60, "labeled": True, "ignore": False, "review_status": "reviewed", "reviewer": "reviewer-1", "revision": 1, "reviewed_at": "2026-09-15T00:00:00Z", "annotation_confidence": 1.0}
    ignored_meta = {**frame_meta, "source_frame": 4, "pts": 240, "ignore": True}
    value = manifest(annotations=[base_annotation], frame_labels=[frame_meta, ignored_meta])
    parsed = load_annotation_manifest(write(tmp_path, value), "abc")
    reference = mot_reference_from_manifest(parsed)
    assert reference["sequences"]["shot-0"]["frames"]["1"]["objects"] == []
    assert reference["sequences"]["shot-0"]["frames"]["4"]["ignore"] is True


def test_unreviewed_manifest_cannot_become_evaluator_reference(tmp_path) -> None:
    value = manifest(reviewed=False)
    parsed = load_annotation_manifest(write(tmp_path, value), "abc", require_reviewed=False)
    with pytest.raises(AnnotationError, match="unreviewed"):
        mot_reference_from_manifest(parsed)


def test_mot_reference_includes_player_tracks_and_keeps_other_labels_in_manifest(tmp_path) -> None:
    base = manifest()["annotations"][0]
    value = manifest(annotations=[
        {**base, "id": "player", "track_id": "p1", "label": "player"},
        {**base, "id": "official", "track_id": "o1", "label": "official", "source_frame": 3},
        {**base, "id": "football", "track_id": "b1", "label": "football", "source_frame": 4},
    ])

    parsed = load_annotation_manifest(write(tmp_path, value), "abc")
    reference = mot_reference_from_manifest(parsed)

    reference_ids = [
        item["id"]
        for frame in reference["sequences"]["shot-0"]["frames"].values()
        for item in frame["objects"]
    ]
    assert reference_ids == ["p1"]
    assert len(parsed.annotations) == 3
