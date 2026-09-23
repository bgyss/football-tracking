from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo, frame_pts


ROOT = Path(__file__).resolve().parents[1]


def make_source(path: Path) -> tuple[dict[str, object], tuple[int, ...]]:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    for frame in range(8):
        image = np.full((24, 32, 3), frame, dtype=np.uint8)
        writer.write(image)
    writer.release()
    info = VideoInfo.from_path(path)
    pts = frame_pts(path)
    if len(pts) != info.frame_count:
        pts = tuple(round(frame * info.time_base[1] / (info.source_fps * info.time_base[0])) for frame in range(info.frame_count))
    return {
        "sha256": sha256_file(path),
        "width": info.width,
        "height": info.height,
        "frame_count": info.frame_count,
        "time_base": list(info.time_base),
    }, pts


def make_run(tmp_path: Path) -> tuple[Path, Path, dict[str, object], tuple[int, ...], dict[str, str]]:
    source_path = tmp_path / "source.mp4"
    source, pts = make_source(source_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    row = {
        "run_id": "run-1",
        "shot_id": "shot-0",
        "frame_index": "3",
        "pts": str(pts[3]),
        "time_base": json.dumps(source["time_base"]),
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
    with (run_dir / "observations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (run_dir / "run-manifest.json").write_text(json.dumps({
        "run_id": "run-1",
        "input_sha256": source["sha256"],
        "input_duration_s": 0.8,
        "source_fps": 10.0,
        "frame_count": 8,
        "detector": "synthetic_proxy",
        "detector_version": "proxy",
        "tracker": "botsort",
        "tracker_version": "test",
        "config_hash": "cfg-1",
    }), encoding="utf-8")
    (run_dir / "shots.json").write_text(json.dumps({"ranges": [[0, 8]], "boundaries": [{"frame_index": 0, "pts": pts[0], "reason": "start", "confidence": 1.0}]}), encoding="utf-8")
    return source_path, run_dir, source, pts, row


def run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT, env=env, capture_output=True, text=True, check=False)


def test_export_cvat_keeps_source_frames_and_inference_provenance(tmp_path) -> None:
    source_path, run_dir, source, pts, raw_row = make_run(tmp_path)
    output_dir = tmp_path / "cvat"

    result = run_script("export_cvat.py", "--run-dir", str(run_dir), "--source", str(source_path), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr
    root = ET.parse(output_dir / "annotations.xml").getroot()
    task = root.find("./meta/task")
    assert task is not None
    original_size = task.find("original_size")
    assert original_size is not None
    assert (original_size.findtext("width"), original_size.findtext("height")) == ("32", "24")
    assert root.findall("tag") == []
    label_types = {label.findtext("name"): label.findtext("type") for label in task.findall("./labels/label")}
    assert label_types["player"] == "bbox"
    assert label_types["calibration_landmark"] == "points"
    assert label_types["timing_event"] == "points"
    player_label = next(label for label in task.findall("./labels/label") if label.findtext("name") == "player")
    mutability = {
        attribute.findtext("name"): attribute.findtext("mutable")
        for attribute in player_label.findall("./attributes/attribute")
    }
    assert mutability["source_run_id"] == "False"
    assert mutability["source_shot_id"] == "False"
    assert mutability["source_pts"] == "True"
    track = root.find("track")
    assert track is not None and track.attrib["label"] == "player"
    box = track.find("box")
    assert box is not None and box.attrib["frame"] == "3"
    attributes = {node.attrib["name"]: node.text for node in box.findall("attribute")}
    assert attributes["source_pts"] == str(pts[3])
    assert attributes["detection_score"] == "0.91"
    assert attributes["source_tracklet_id"] == "shot-0:t7"

    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source"] == source
    assert provenance["run"]["run_id"] == "run-1"
    assert provenance["observation_provenance"][0]["raw_row"] == raw_row


def test_export_cvat_uses_effective_run_shot_segments(tmp_path) -> None:
    source_path, run_dir, _source, _pts, _raw_row = make_run(tmp_path)
    (run_dir / "shots.json").write_text(json.dumps({
        "ranges": [[0, 8]],
        "segments": [{"shot_id": "wide-sideline", "source_shot_id": "shot-0", "start_frame": 0, "end_frame": 8}],
        "boundaries": [{"frame_index": 0, "pts": 0, "reason": "start", "confidence": 1.0}],
    }), encoding="utf-8")
    output_dir = tmp_path / "cvat-segments"

    result = run_script("export_cvat.py", "--run-dir", str(run_dir), "--source", str(source_path), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr
    root = ET.parse(output_dir / "annotations.xml").getroot()
    box = root.find("./track/box")
    assert box is not None
    attributes = {node.attrib["name"]: node.text for node in box.findall("attribute")}
    assert attributes["shot_id"] == "wide-sideline"
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["shots"]["wide-sideline"]["source_shot_id"] == "shot-0"
    assert provenance["observation_provenance"][0]["source_shot_id"] == "shot-0"


def test_import_cvat_builds_reviewed_player_reference_and_keeps_other_labels(tmp_path) -> None:
    source_path, run_dir, source, pts, raw_row = make_run(tmp_path)
    provenance = {
        "schema_version": 1,
        "source": source,
        "run": {"run_id": "run-1", "detector": "synthetic_proxy", "tracker": "botsort", "config_hash": "cfg-1"},
        "shots": {"reviewed-segment": {"source_shot_id": "shot-0", "start_frame": 1, "end_frame": 8, "start_pts": pts[1], "play_id": "play-1", "split": "development", "camera_label": "sideline"}},
        "observation_provenance": [{"shot_id": "shot-0", "source_shot_id": "shot-0", "source_frame": 3, "tracklet_id": "shot-0:t7", "raw_row": raw_row}],
    }
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    xml_path = tmp_path / "reviewed.xml"
    xml_path.write_text(f'''<annotations>
      <track id="10" label="player" source="manual"><box frame="3" xtl="2" ytl="3" xbr="12" ybr="21" outside="0" occluded="1" keyframe="1">
        <attribute name="source_pts">{pts[3]}</attribute><attribute name="source_tracklet_id">shot-0:t7</attribute>
        <attribute name="source_run_id">run-1</attribute><attribute name="source_shot_id">shot-0</attribute>
        <attribute name="anonymous_id">anon-12</attribute><attribute name="team">DET</attribute>
        <attribute name="visibility">visible</attribute><attribute name="detection_score">0.91</attribute><attribute name="review_status">reviewed</attribute>
      </box></track>
      <track id="11" label="official"><box frame="4" xtl="14" ytl="2" xbr="20" ybr="22" outside="0" occluded="0" keyframe="1"><attribute name="source_pts">{pts[3]}</attribute><attribute name="visibility">visible</attribute><attribute name="review_status">accepted</attribute></box></track>
      <track id="12" label="football"><box frame="5" xtl="20" ytl="10" xbr="22" ybr="12" outside="0" occluded="0" keyframe="1"><attribute name="source_pts">{pts[5]}</attribute><attribute name="review_status">reviewed</attribute></box></track>
      <track id="13" label="timing_event"><points frame="3" points="0,0" outside="0" occluded="0" keyframe="1"><attribute name="event">snap</attribute><attribute name="source_pts">{pts[3]}</attribute><attribute name="play_id">play-1</attribute><attribute name="correspondence_id">snap</attribute><attribute name="play_time_s">0</attribute><attribute name="review_status">reviewed</attribute></points><points frame="4" points="0,0" outside="0" occluded="0" keyframe="0"><attribute name="event">contact</attribute><attribute name="source_pts">{pts[4]}</attribute><attribute name="play_id">play-1</attribute><attribute name="correspondence_id">contact</attribute><attribute name="play_time_s">1</attribute><attribute name="review_status">reviewed</attribute></points></track>
      <track id="14" label="calibration_landmark"><points frame="2" points="4,5" outside="0" occluded="0" keyframe="1"><attribute name="source_pts">{pts[2]}</attribute><attribute name="landmark_id">yardline:20:sideline:near</attribute><attribute name="field_x_yards">20</attribute><attribute name="field_y_yards">0</attribute><attribute name="role">fit</attribute><attribute name="review_status">reviewed</attribute></points><points frame="4" points="5,6" outside="0" occluded="0" keyframe="0"><attribute name="source_pts">{pts[4]}</attribute><attribute name="landmark_id">yardline:20:sideline:near</attribute><attribute name="field_x_yards">20</attribute><attribute name="field_y_yards">0</attribute><attribute name="role">fit</attribute><attribute name="review_status">reviewed</attribute></points></track>
      <track id="15" label="player"><box frame="6" xtl="2" ytl="3" xbr="12" ybr="21" outside="0" occluded="0" keyframe="1"><attribute name="source_pts">{pts[6]}</attribute><attribute name="review_status">rejected</attribute></box></track>
    </annotations>''', encoding="utf-8")
    output_dir = tmp_path / "imported"

    result = run_script(
        "import_cvat.py", "--annotations", str(xml_path), "--provenance", str(provenance_path),
        "--source", str(source_path), "--output-dir", str(output_dir), "--reviewed",
        "--reviewer", "reviewer-1", "--reviewed-at", "2026-09-22T12:00:00Z",
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "annotations.json").read_text(encoding="utf-8"))
    assert manifest["reviewed"] is True
    assert manifest["shots"]["reviewed-segment"]["start_frame"] == 1
    assert {item["label"] for item in manifest["annotations"]} == {"player", "official", "football"}
    assert len(manifest["annotations"]) == 3
    player = next(item for item in manifest["annotations"] if item["label"] == "player")
    assert player["source_frame"] == 3
    assert player["pts"] == pts[3]
    assert player["global_id"] == "anon-12"
    assert player["team"] == "DET"
    assert player["visibility"] == "occluded"
    assert player["inference_provenance"]["raw_row"] == raw_row
    assert len(manifest["timing_events"]) == 1
    assert manifest["timing_events"][0]["event"] == "snap"
    assert manifest["timing_events"][0]["play_time_s"] == 0.0
    assert len(manifest["landmarks"]) == 1
    assert manifest["landmarks"][0]["landmark_id"] == "yardline:20:sideline:near"

    reference = json.loads((output_dir / "mot-reference.json").read_text(encoding="utf-8"))
    objects = reference["sequences"]["reviewed-segment"]["frames"]["3"]["objects"]
    assert [item["id"] for item in objects] == [player["track_id"]]
    assert reference["sequences"]["reviewed-segment"]["frames"].get("6", {}).get("objects", []) == []
    assert reference["cross_shot_identity"]["reviewed-segment"][player["track_id"]] == "anon-12"


def test_import_cvat_refuses_to_mark_tracks_reviewed_without_explicit_flag(tmp_path) -> None:
    source_path, _run_dir, source, _pts, _raw_row = make_run(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps({
        "schema_version": 1,
        "source": source,
        "shots": {"shot-0": {"start_frame": 0, "end_frame": 8, "play_id": None, "split": "unassigned", "camera_label": "unknown"}},
        "observation_provenance": [],
    }), encoding="utf-8")
    xml_path = tmp_path / "empty.xml"
    xml_path.write_text("<annotations />", encoding="utf-8")

    result = run_script(
        "import_cvat.py", "--annotations", str(xml_path), "--provenance", str(provenance_path),
        "--source", str(source_path), "--output-dir", str(tmp_path / "unreviewed"),
        "--reviewer", "reviewer-1", "--reviewed-at", "2026-09-22T12:00:00Z",
    )

    assert result.returncode != 0
    assert "--reviewed" in result.stderr
    assert not (tmp_path / "unreviewed" / "mot-reference.json").exists()


def test_import_cvat_refuses_unreviewed_shapes_even_with_global_review_assertion(tmp_path) -> None:
    source_path, _run_dir, source, pts, _raw_row = make_run(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps({
        "schema_version": 1,
        "source": source,
        "shots": {"shot-0": {"start_frame": 0, "end_frame": 8, "play_id": None, "split": "unassigned", "camera_label": "unknown"}},
        "observation_provenance": [],
    }), encoding="utf-8")
    xml_path = tmp_path / "unreviewed-shape.xml"
    xml_path.write_text(f'''<annotations><track id="1" label="player"><box frame="2" xtl="1" ytl="1" xbr="8" ybr="12" outside="0" occluded="0" keyframe="1"><attribute name="source_pts">{pts[2]}</attribute><attribute name="review_status">unreviewed</attribute></box></track></annotations>''', encoding="utf-8")
    output_dir = tmp_path / "rejected-import"

    result = run_script(
        "import_cvat.py", "--annotations", str(xml_path), "--provenance", str(provenance_path),
        "--source", str(source_path), "--output-dir", str(output_dir), "--reviewed",
        "--reviewer", "reviewer-1", "--reviewed-at", "2026-09-22T12:00:00Z",
    )

    assert result.returncode != 0
    assert "unreviewed" in result.stderr
    assert not (output_dir / "mot-reference.json").exists()


def test_import_cvat_rejects_provenance_from_a_different_run(tmp_path) -> None:
    source_path, _run_dir, source, pts, raw_row = make_run(tmp_path)
    mismatched_row = {**raw_row, "run_id": "run-2"}
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps({
        "schema_version": 1,
        "source": source,
        "run": {"run_id": "run-2"},
        "shots": {"shot-0": {"start_frame": 0, "end_frame": 8, "play_id": None, "split": "unassigned", "camera_label": "unknown"}},
        "observation_provenance": [{"shot_id": "shot-0", "source_shot_id": "shot-0", "source_frame": 3, "tracklet_id": "shot-0:t7", "raw_row": mismatched_row}],
    }), encoding="utf-8")
    xml_path = tmp_path / "wrong-run.xml"
    xml_path.write_text(f'''<annotations><track id="1" label="player"><box frame="3" xtl="1" ytl="1" xbr="8" ybr="12" outside="0" occluded="0" keyframe="1"><attribute name="source_pts">{pts[3]}</attribute><attribute name="source_run_id">run-1</attribute><attribute name="source_shot_id">shot-0</attribute><attribute name="source_tracklet_id">shot-0:t7</attribute><attribute name="review_status">reviewed</attribute></box></track></annotations>''', encoding="utf-8")
    output_dir = tmp_path / "wrong-run-output"

    result = run_script(
        "import_cvat.py", "--annotations", str(xml_path), "--provenance", str(provenance_path),
        "--source", str(source_path), "--output-dir", str(output_dir), "--reviewed",
        "--reviewer", "reviewer-1", "--reviewed-at", "2026-09-22T12:00:00Z",
    )

    assert result.returncode != 0
    assert "run" in result.stderr
    assert not (output_dir / "mot-reference.json").exists()


def test_import_cvat_rejects_partial_source_provenance_fields(tmp_path) -> None:
    source_path, _run_dir, source, pts, raw_row = make_run(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps({
        "schema_version": 1,
        "source": source,
        "run": {"run_id": "run-1"},
        "shots": {"shot-0": {"start_frame": 0, "end_frame": 8, "play_id": None, "split": "unassigned", "camera_label": "unknown"}},
        "observation_provenance": [{"shot_id": "shot-0", "source_shot_id": "shot-0", "source_frame": 3, "tracklet_id": "shot-0:t7", "raw_row": raw_row}],
    }), encoding="utf-8")
    xml_path = tmp_path / "partial-provenance.xml"
    xml_path.write_text(f'''<annotations><track id="1" label="player"><box frame="3" xtl="1" ytl="1" xbr="8" ybr="12" outside="0" occluded="0" keyframe="1"><attribute name="source_pts">{pts[3]}</attribute><attribute name="source_tracklet_id">shot-0:t7</attribute><attribute name="review_status">reviewed</attribute></box></track></annotations>''', encoding="utf-8")
    output_dir = tmp_path / "partial-provenance-output"

    result = run_script(
        "import_cvat.py", "--annotations", str(xml_path), "--provenance", str(provenance_path),
        "--source", str(source_path), "--output-dir", str(output_dir), "--reviewed",
        "--reviewer", "reviewer-1", "--reviewed-at", "2026-09-22T12:00:00Z",
    )

    assert result.returncode != 0
    assert "provenance" in result.stderr
    assert not (output_dir / "mot-reference.json").exists()
