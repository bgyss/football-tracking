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

    # Minimal calibration that maps frame corners to field coordinates
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
    # In a two-shot scenario, unresolved_shots should be empty
    assert report.get("unresolved_shots") == []
    assert report.get("note") == "only the first two aligned shots are resolved in this milestone"
