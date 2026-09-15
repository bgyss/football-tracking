from __future__ import annotations

import cv2
import json
import numpy as np
import csv

from football_tracking.cli import build_parser, main


def make_video(path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    for index in range(4):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        cv2.rectangle(frame, (2 + index, 2), (8 + index, 12), (255, 0, 0), -1)
        writer.write(frame)
    writer.release()


def test_parser_exposes_run_and_benchmark_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["run", "--input", "video.mp4", "--output", "out"]).command == "run"
    assert parser.parse_args(["benchmark", "--input", "video.mp4", "--output", "out"]).command == "benchmark"
    assert parser.parse_args(["cache", "--input", "video.mp4", "--output", "chunk.jsonl", "--detector-class-mapping", "classes.json"]).command == "cache"


def test_parser_exposes_explicit_mcbyte_mask_configuration() -> None:
    args = build_parser().parse_args([
        "run", "--input", "video.mp4", "--output", "out", "--tracker", "mcbyte",
        "--mcbyte-device", "cpu", "--mcbyte-sam-checkpoint", "sam.pth",
        "--mcbyte-cutie-checkpoint", "cutie.pth", "--mcbyte-masks", "off",
        "--detector-class-mapping", "classes.json",
    ])

    assert args.tracker == "mcbyte"
    assert args.mcbyte_device == "cpu"
    assert args.mcbyte_masks == "off"
    assert args.detector_class_mapping.name == "classes.json"


def test_benchmark_writes_report_for_synthetic_detector(tmp_path) -> None:
    input_path = tmp_path / "tiny.mp4"
    output_path = tmp_path / "report"
    make_video(input_path)

    exit_code = main(["benchmark", "--input", str(input_path), "--output", str(output_path), "--detector", "synthetic", "--tracker", "iou", "--manual-cut", "2"])

    assert exit_code == 0
    assert (output_path / "benchmark.json").is_file()
    report = (output_path / "benchmark.json").read_text()
    assert '"decode"' in report
    assert '"status": "complete"' in report


def test_run_rejects_missing_input() -> None:
    assert main(["run", "--input", "missing.mp4", "--output", "out", "--detector", "synthetic"]) != 0


def test_run_rejects_a_nonpositive_memory_budget(tmp_path) -> None:
    input_path = tmp_path / "tiny.mp4"
    make_video(input_path)

    assert main(["run", "--input", str(input_path), "--output", str(tmp_path / "out"), "--detector", "synthetic", "--max-memory-mb", "0"]) != 0


def test_cache_chunks_merge_into_a_strict_complete_replay_cache(tmp_path) -> None:
    input_path = tmp_path / "tiny.mp4"
    make_video(input_path)
    classes = tmp_path / "classes.json"
    classes.write_text('{"0":"player"}', encoding="utf-8")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    merged = tmp_path / "merged.jsonl"
    common = ["--input", str(input_path), "--detector", "synthetic", "--detector-class-mapping", str(classes)]

    assert main(["cache", *common, "--output", str(first), "--start-frame", "0", "--end-frame", "2"]) == 0
    assert main(["cache", *common, "--output", str(second), "--start-frame", "2", "--end-frame", "4"]) == 0
    assert main(["merge-cache", *common, "--output", str(merged), "--part", str(first), "--part", str(second)]) == 0
    assert merged.is_file()


def test_run_records_detection_and_tracking_stage_timings(tmp_path) -> None:
    input_path = tmp_path / "timed.mp4"
    output_path = tmp_path / "run"
    make_video(input_path)

    assert main(["run", "--input", str(input_path), "--output", str(output_path), "--detector", "synthetic", "--tracker", "iou", "--manual-cut", "2"]) == 0
    metrics = (output_path / "metrics.json").read_text()
    assert '"detection_s"' in metrics
    assert '"tracking_s"' in metrics
    with (output_path / "observations.csv").open(newline="") as handle:
        assert {row["shot_id"] for row in csv.DictReader(handle)} == {"shot-0", "shot-1"}


def test_run_emits_identity_links_evidence_and_defaults_to_unlinked(tmp_path) -> None:
    input_path = tmp_path / "tiny.mp4"
    output = tmp_path / "run"
    make_video(input_path)

    exit_code = main([
        "run", "--input", str(input_path), "--output", str(output),
        "--detector", "synthetic", "--tracker", "iou", "--manual-cut", "2",
    ])
    assert exit_code == 0

    report = json.loads((output / "identity-links.json").read_text(encoding="utf-8"))
    assert report["status"] == "not_attempted"
    assert report["reason"] == "--play-alignment was not provided"
    assert report["links"] == []
    assert report["cross_shot_player_ids"] == 0
    assert report["player_id_count"] == report["tracklet_count"]


def test_run_abstains_without_calibration_and_records_the_reason(tmp_path) -> None:
    input_path = tmp_path / "tiny.mp4"
    make_video(input_path)

    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 0, "event": "snap"},
            {"shot_id": "shot-1", "source_frame": 2, "event": "snap"},
        ],
    }), encoding="utf-8")

    output = tmp_path / "run"
    exit_code = main([
        "run", "--input", str(input_path), "--output", str(output),
        "--detector", "synthetic", "--tracker", "iou", "--manual-cut", "2",
        "--play-alignment", str(alignment),
    ])
    assert exit_code == 0

    report = json.loads((output / "identity-links.json").read_text(encoding="utf-8"))
    assert report["status"] == "abstained"
    assert report["reason"] == "no calibrated field positions for the aligned shots"
    assert report["cross_shot_player_ids"] == 0

    review = json.loads((output / "review.json").read_text(encoding="utf-8"))
    assert review["cross_view_identity_resolved"] is False

    # Parse with DictReader like every other CSV assertion in this file: the
    # time_base column precedes play_id and serialises as "[1,10240]", so a
    # naive split(",") on a data row shifts every later column by one.
    with (output / "observations.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["play_id"] for row in rows} == {"play-1"}


def test_run_rejects_unreviewed_play_alignment(tmp_path) -> None:
    input_path = tmp_path / "tiny.mp4"
    make_video(input_path)

    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({"play_id": "play-1", "anchors": []}), encoding="utf-8")

    exit_code = main([
        "run", "--input", str(input_path), "--output", str(tmp_path / "run"),
        "--detector", "synthetic", "--tracker", "iou", "--manual-cut", "2",
        "--play-alignment", str(alignment),
    ])
    assert exit_code == 2


def test_run_records_unresolved_shots_in_two_shot_cross_identity(tmp_path) -> None:
    input_path = tmp_path / "tiny.mp4"
    make_video(input_path)

    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 0, "event": "snap"},
            {"shot_id": "shot-1", "source_frame": 2, "event": "snap"},
        ],
    }), encoding="utf-8")

    # Shot-specific calibration (both shots mapped explicitly, not via the "*"
    # shared fallback) that maps frame corners to field coordinates.
    landmarks = {
        "image_points": [[0, 0], [32, 0], [32, 24], [0, 24]],
        "field_points": [[0, 0], [120, 0], [120, 53.33], [0, 53.33]],
    }
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"shots": {"shot-0": landmarks, "shot-1": landmarks}}), encoding="utf-8")

    output = tmp_path / "run"
    exit_code = main([
        "run", "--input", str(input_path), "--output", str(output),
        "--detector", "synthetic", "--tracker", "iou", "--manual-cut", "2",
        "--play-alignment", str(alignment),
        "--calibration", str(calibration),
    ])
    assert exit_code == 0

    report = json.loads((output / "identity-links.json").read_text(encoding="utf-8"))
    # In a two-shot scenario, unresolved_shots should be empty
    assert report.get("unresolved_shots") == []
    assert report.get("note") == "only the first two aligned shots are resolved in this milestone"


def test_run_abstains_on_a_shared_homography_instead_of_merging_on_it(tmp_path) -> None:
    # A calibration file with no `shots` object fits one shared homography (keyed
    # "*"). A homography is a per-camera-pose transform: applying the same one to
    # both shots would make shot-1's "field position" a restatement of its image
    # coordinates, so accepting it as cross-shot evidence would merge identities on
    # image-coordinate proximity across the cut -- exactly what this design forbids.
    input_path = tmp_path / "tiny.mp4"
    make_video(input_path)

    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 0, "event": "snap"},
            {"shot_id": "shot-1", "source_frame": 2, "event": "snap"},
        ],
    }), encoding="utf-8")

    # No "shots" object: load_calibrations returns a single shared {"*": ...} entry.
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "image_points": [[0, 0], [32, 0], [32, 24], [0, 24]],
        "field_points": [[0, 0], [120, 0], [120, 53.33], [0, 53.33]],
    }), encoding="utf-8")

    output = tmp_path / "run"
    exit_code = main([
        "run", "--input", str(input_path), "--output", str(output),
        "--detector", "synthetic", "--tracker", "iou", "--manual-cut", "2",
        "--play-alignment", str(alignment),
        "--calibration", str(calibration),
    ])
    assert exit_code == 0

    report = json.loads((output / "identity-links.json").read_text(encoding="utf-8"))
    assert report["status"] == "abstained"
    assert report["reason"] == "cross-shot resolution requires shot-specific calibration"
    assert report["cross_shot_player_ids"] == 0
    assert report["player_id_count"] == report["tracklet_count"]
    assert "accepted_links" not in report

    review = json.loads((output / "review.json").read_text(encoding="utf-8"))
    assert review["cross_view_identity_resolved"] is False

    manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete_with_unresolved"


REPLAY_WIDTH, REPLAY_HEIGHT = 100, 60


def make_replay_video(path, frames_per_shot: int) -> None:
    """A wide video whose two halves are solid, distinct colors (team evidence)
    and whose detections (from SyntheticDetector's two fixed, frame-content
    independent boxes) are static -- so a shot-0/shot-1 split of this same content
    models a play observed twice from different cameras, once per shot. The frame
    is wide enough that each fixed detection box (8% of frame width) crops to at
    least 4px, the minimum team_feature_from_crop requires."""

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (REPLAY_WIDTH, REPLAY_HEIGHT))
    assert writer.isOpened()
    frame = np.zeros((REPLAY_HEIGHT, REPLAY_WIDTH, 3), dtype=np.uint8)
    frame[:, : REPLAY_WIDTH // 2] = (255, 0, 0)  # BGR -> RGB (0, 0, 1): "blue" team half
    frame[:, REPLAY_WIDTH // 2 :] = (0, 255, 0)  # BGR -> RGB (0, 1, 0): "green" team half
    for _ in range(frames_per_shot * 2):
        writer.write(frame)
    writer.release()


def test_run_reaches_a_resolved_cross_shot_outcome_from_view_invariant_evidence(tmp_path) -> None:
    frames_per_shot = 7
    input_path = tmp_path / "replay.mp4"
    make_replay_video(input_path, frames_per_shot)

    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 0, "event": "snap"},
            {"shot_id": "shot-1", "source_frame": frames_per_shot, "event": "snap"},
        ],
    }), encoding="utf-8")

    landmarks = {
        "image_points": [[0, 0], [REPLAY_WIDTH, 0], [REPLAY_WIDTH, REPLAY_HEIGHT], [0, REPLAY_HEIGHT]],
        "field_points": [[0, 0], [120, 0], [120, 53.33], [0, 53.33]],
    }
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"shots": {"shot-0": landmarks, "shot-1": landmarks}}), encoding="utf-8")

    # Forces real (non-"unknown") team labels onto the synthetic fixture's crops,
    # which is required for cross_shot_candidate_scores to consider a pair at all.
    prototypes = tmp_path / "prototypes.json"
    prototypes.write_text(json.dumps({"blue": [0.0, 0.0, 1.0], "green": [0.0, 1.0, 0.0]}), encoding="utf-8")

    output = tmp_path / "run"
    exit_code = main([
        "run", "--input", str(input_path), "--output", str(output),
        "--detector", "synthetic", "--tracker", "iou", "--manual-cut", str(frames_per_shot),
        "--play-alignment", str(alignment),
        "--calibration", str(calibration),
        "--team-prototypes", str(prototypes),
    ])
    assert exit_code == 0

    report = json.loads((output / "identity-links.json").read_text(encoding="utf-8"))
    # Asserted unconditionally: the fixture is deterministic (synthetic detector,
    # fixed geometry), so a branch-tolerant assertion here would let a regression
    # slide silently into the abstain path.
    assert report["status"] == "resolved"
    assert report["candidate_pairs"] > 0
    assert report["accepted_links"] > 0
    assert report["cross_shot_player_ids"] > 0
    assert {link["decision"] for link in report["links"]} == {"same"}
    assert report["player_id_count"] < report["tracklet_count"]

    review = json.loads((output / "review.json").read_text(encoding="utf-8"))
    assert review["status"] == "resolved_cross_view"
    assert review["cross_view_identity_resolved"] is True

    manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
