from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from football_tracking.cvat import CvatBox, CvatError, CvatPoint, CvatTrack, TaskFrameMap, write_cvat_video_xml
from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo, frame_pts
from scripts.export_cvat import export_observations
from scripts.import_cvat import import_cvat


def make_run(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str], tuple[int, ...]]:
    source_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    for frame in range(8):
        writer.write(np.full((24, 32, 3), frame, dtype=np.uint8))
    writer.release()
    info = VideoInfo.from_path(source_path)
    pts_values = frame_pts(source_path)
    assert len(pts_values) == info.frame_count

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    observations_path = run_dir / "observations.csv"
    row = {
        "run_id": "run-1",
        "shot_id": "shot-0",
        "frame_index": "3",
        "pts": str(pts_values[3]),
        "time_base": json.dumps(list(info.time_base)),
        "tracklet_id": "shot-0:t7",
        "player_id": "proposal-player-1",
        "play_id": "play-1",
        "bbox_xyxy_px": json.dumps([2.0, 3.0, 12.0, 21.0]),
        "detection_score": "0.91",
        "team": "DET",
        "team_score": "0.8",
        "jersey_number": "",
        "field_xy_yards": "",
        "position_source": "",
        "calibration_id": "",
        "identity_version": "1",
        "calibration_status": "",
        "calibration_reason": "",
        "position_uncertainty_yards": "",
        "play_time_s": "",
        "time_map_id": "",
        "source_tracklet_id": "",
    }
    with observations_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return source_path, observations_path, run_dir, row, pts_values


def export_bundle(tmp_path: Path, source_path: Path, observations_path: Path) -> tuple[Path, Path, TaskFrameMap]:
    bundle = tmp_path / "cvat.zip"
    frame_map_path = tmp_path / "task-frame-map.json"
    export_observations(
        observations_path,
        source_path,
        shot_id="shot-0",
        start_frame=2,
        end_frame=7,
        output_path=bundle,
        frame_map_output=frame_map_path,
    )
    frame_map = TaskFrameMap.from_dict(json.loads(frame_map_path.read_text(encoding="utf-8")))
    return bundle, frame_map_path, frame_map


def test_export_observations_keeps_source_map_and_raw_inference_row(tmp_path) -> None:
    source_path, observations_path, _run_dir, raw_row, pts = make_run(tmp_path)
    bundle, frame_map_path, frame_map = export_bundle(tmp_path, source_path, observations_path)
    assert bundle.is_file()
    assert frame_map_path.is_file()
    assert frame_map.frames[1].source_frame == 3
    assert frame_map.frames[1].source_pts == pts[3]

    from zipfile import ZipFile

    with ZipFile(bundle) as archive:
        xml = archive.read("annotations.xml")
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    label_nodes = root.findall("./meta/task/labels/label")
    labels = {item.findtext("name") for item in label_nodes}
    assert {"player", "official", "football", "timing_event", "field_landmark"} <= labels
    attributes_by_label = {
        item.findtext("name"): {attribute.findtext("name") for attribute in item.findall("./attributes/attribute")}
        for item in label_nodes
    }
    assert {"team", "visibility", "review_status"} <= attributes_by_label["official"]
    assert {"visibility", "review_status"} <= attributes_by_label["football"]
    player = next(track for track in root.findall("track") if track.attrib["label"] == "player")
    box = player.find("box")
    assert box is not None and box.attrib["frame"] == "1"
    box_attributes = {item.attrib["name"]: item.text for item in box.findall("attribute")}
    assert box_attributes["source_frame"] == "3"
    assert box_attributes["source_pts"] == str(pts[3])
    assert json.loads(box_attributes["source_inference_row_json"]) == raw_row
    assert player.find("attribute[@name='source_run_id']").text == "run-1"


def test_reviewed_cvat_tracks_create_manifest_and_player_only_mot_reference(tmp_path) -> None:
    source_path, observations_path, _run_dir, raw_row, pts = make_run(tmp_path)
    _bundle, frame_map_path, frame_map = export_bundle(tmp_path, source_path, observations_path)
    raw_row_json = json.dumps(raw_row, sort_keys=True, separators=(",", ":"))
    identity_review = {
        "cross_shot_review_status": "approved",
        "identity_second_reviewer": "reviewer-2",
        "identity_second_reviewed_at": "2026-09-22T12:10:00Z",
        "identity_second_revision": "1",
        "identity_second_confidence": "1.0",
    }
    tracks = (
        CvatTrack(
            0,
            "player",
            "manual",
            {"tracklet_id": "shot-0:t7", "source_run_id": "run-1", "source_shot_id": "shot-0", "source_sha256": frame_map.source_sha256, "global_id": "anon-12", "team": "DET", "review_status": "reviewed", **identity_review},
            (
                CvatBox(1, (2.0, 3.0, 12.0, 21.0), occluded=True, attributes={"source_frame": "3", "source_pts": str(pts[3]), "detection_score": "0.91", "team_score": "0.8", "source_inference_row_json": raw_row_json, "visibility": "visible", "review_status": "reviewed"}),
                CvatBox(2, (3.0, 4.0, 13.0, 22.0), keyframe=False, attributes={"source_frame": "3", "source_pts": str(pts[3]), "source_inference_row_json": raw_row_json, "visibility": "visible", "review_status": "reviewed"}),
            ),
        ),
        CvatTrack(1, "official", "manual", {"review_status": "accepted"}, (CvatBox(2, (14.0, 2.0, 20.0, 22.0), attributes={"source_frame": "4", "source_pts": str(pts[4]), "visibility": "visible", "review_status": "accepted"}),)),
        CvatTrack(2, "football", "manual", {"review_status": "reviewed"}, (CvatBox(3, (20.0, 10.0, 22.0, 12.0), attributes={"source_frame": "5", "source_pts": str(pts[5]), "review_status": "reviewed"}),)),
        CvatTrack(3, "player", "manual", {"review_status": "rejected"}, (CvatBox(4, (2.0, 3.0, 12.0, 21.0), attributes={"source_frame": "6", "source_pts": str(pts[6]), "review_status": "rejected"}),)),
        CvatTrack(6, "player", "manual", {"review_status": "unreviewed"}, (CvatBox(3, None, outside=True, attributes={"source_frame": "5", "source_pts": str(pts[5]), "review_status": "unreviewed"}),)),
        CvatTrack(
            4,
            "timing_event",
            "manual",
            {"review_status": "reviewed"},
            (),
            (
                CvatPoint(1, ((0.0, 0.0),), attributes={"event": "snap", "play_id": "play-1", "correspondence_id": "snap", "play_time_s": "0", "source_frame": "3", "source_pts": str(pts[3]), "review_status": "reviewed"}),
                CvatPoint(2, ((0.0, 0.0),), keyframe=False, attributes={"event": "contact", "play_id": "play-1", "correspondence_id": "contact", "play_time_s": "1", "source_frame": "3", "source_pts": str(pts[3]), "review_status": "reviewed"}),
            ),
        ),
        CvatTrack(5, "field_landmark", "manual", {"review_status": "reviewed"}, (), (CvatPoint(0, ((4.0, 5.0),), attributes={"landmark_id": "yardline:20:sideline:near", "field_x_yards": "20", "field_y_yards": "0", "role": "fit", "source_frame": "2", "source_pts": str(pts[2]), "review_status": "reviewed"}),)),
    )
    reviewed_xml = tmp_path / "reviewed.xml"
    reviewed_xml.write_bytes(write_cvat_video_xml(tracks, frame_map))
    manifest_path = tmp_path / "reviewed-annotations.json"
    reference_path = tmp_path / "mot-reference.json"

    import_cvat(
        reviewed_xml,
        source_path,
        manifest_path,
        shot={"shot_id": "shot-0", "start_frame": 2, "end_frame": 7, "camera_label": "sideline", "play_id": "play-1", "split": "development"},
        frame_map_path=frame_map_path,
        mark_reviewed=True,
        reviewer="reviewer-1",
        revision=1,
        reviewed_at="2026-09-22T12:00:00Z",
        annotation_confidence=1.0,
        mot_reference_output=reference_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {record["label"] for record in manifest["annotations"]} == {"player", "official", "football"}
    assert len(manifest["annotations"]) == 4
    player = next(record for record in manifest["annotations"] if record["label"] == "player")
    assert player["source_frame"] == 3
    assert player["pts"] == pts[3]
    assert player["global_id"] == "anon-12"
    assert player["visibility"] == "occluded"
    assert player["inference_provenance"]["raw_row"] == raw_row
    interpolated = next(record for record in manifest["annotations"] if record["label"] == "player" and record["source_frame"] == 4)
    assert interpolated["pts"] == pts[4]
    assert "inference_provenance" not in interpolated
    assert len(manifest["timing_events"]) == 1
    assert manifest["timing_events"][0]["event"] == "snap"
    assert len(manifest["landmarks"]) == 1

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    objects = reference["sequences"]["shot-0"]["frames"]["3"]["objects"]
    assert [item["id"] for item in objects] == ["shot-0:t7"]
    assert [item["id"] for item in reference["sequences"]["shot-0"]["frames"]["4"]["objects"]] == ["shot-0:t7"]
    assert reference["sequences"]["shot-0"]["frames"].get("6", {}).get("objects", []) == []


def test_reviewed_cvat_import_rejects_unreviewed_shape(tmp_path) -> None:
    source_path, observations_path, _run_dir, _raw_row, pts = make_run(tmp_path)
    _bundle, frame_map_path, frame_map = export_bundle(tmp_path, source_path, observations_path)
    xml_path = tmp_path / "unreviewed.xml"
    track = CvatTrack(0, "player", "auto", {"tracklet_id": "shot-0:t7"}, (CvatBox(1, (2.0, 3.0, 12.0, 21.0), attributes={"source_frame": "3", "source_pts": str(pts[3]), "review_status": "unreviewed"}),))
    xml_path.write_bytes(write_cvat_video_xml((track,), frame_map))

    with pytest.raises(CvatError, match="reviewed or accepted"):
        import_cvat(
            xml_path,
            source_path,
            tmp_path / "unreviewed.json",
            shot={"shot_id": "shot-0", "start_frame": 2, "end_frame": 7, "camera_label": "sideline", "play_id": None, "split": "development"},
            frame_map_path=frame_map_path,
            mark_reviewed=True,
            reviewer="reviewer-1",
            revision=1,
            reviewed_at="2026-09-22T12:00:00Z",
            annotation_confidence=1.0,
        )


def test_reviewed_cvat_import_rejects_run_mismatch_in_raw_row(tmp_path) -> None:
    source_path, observations_path, _run_dir, raw_row, pts = make_run(tmp_path)
    _bundle, frame_map_path, frame_map = export_bundle(tmp_path, source_path, observations_path)
    mismatched_row = {**raw_row, "run_id": "other-run"}
    track = CvatTrack(
        0,
        "player",
        "auto",
        {"tracklet_id": "shot-0:t7", "source_run_id": "run-1", "source_shot_id": "shot-0"},
        (CvatBox(1, (2.0, 3.0, 12.0, 21.0), attributes={"source_frame": "3", "source_pts": str(pts[3]), "source_inference_row_json": json.dumps(mismatched_row), "review_status": "reviewed"}),),
    )
    xml_path = tmp_path / "wrong-run.xml"
    xml_path.write_bytes(write_cvat_video_xml((track,), frame_map))

    with pytest.raises(CvatError, match="run_id"):
        import_cvat(
            xml_path,
            source_path,
            tmp_path / "wrong-run.json",
            shot={"shot_id": "shot-0", "start_frame": 2, "end_frame": 7, "camera_label": "sideline", "play_id": None, "split": "development"},
            frame_map_path=frame_map_path,
            mark_reviewed=True,
            reviewer="reviewer-1",
            revision=1,
            reviewed_at="2026-09-22T12:00:00Z",
            annotation_confidence=1.0,
        )


@pytest.mark.parametrize("missing_lineage", ["source_run_id", "source_shot_id"])
def test_reviewed_cvat_import_requires_raw_row_lineage(tmp_path, missing_lineage) -> None:
    source_path, observations_path, _run_dir, raw_row, pts = make_run(tmp_path)
    _bundle, frame_map_path, frame_map = export_bundle(tmp_path, source_path, observations_path)
    track_attributes = {"tracklet_id": "shot-0:t7", "source_run_id": "run-1", "source_shot_id": "shot-0"}
    track_attributes.pop(missing_lineage)
    track = CvatTrack(
        0,
        "player",
        "auto",
        track_attributes,
        (CvatBox(1, (2.0, 3.0, 12.0, 21.0), attributes={"source_frame": "3", "source_pts": str(pts[3]), "source_inference_row_json": json.dumps(raw_row), "review_status": "reviewed"}),),
    )
    xml_path = tmp_path / "missing-lineage.xml"
    xml_path.write_bytes(write_cvat_video_xml((track,), frame_map))

    with pytest.raises(CvatError, match="requires source_run_id and source_shot_id"):
        import_cvat(
            xml_path,
            source_path,
            tmp_path / "manifest.json",
            shot={"shot_id": "shot-0", "start_frame": 2, "end_frame": 7, "camera_label": "sideline", "play_id": "play-1", "split": "development"},
            frame_map_path=frame_map_path,
            mark_reviewed=True,
            reviewer="reviewer-1",
            revision=1,
            reviewed_at="2026-09-22T12:00:00Z",
            annotation_confidence=1.0,
        )
