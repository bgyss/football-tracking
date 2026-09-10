from __future__ import annotations

import cv2
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
