from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from football_tracking.cli import build_parser, run_pipeline
from football_tracking.metrics import sha256_file


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    for frame in range(12):
        writer.write(np.full((48, 64, 3), frame * 12, dtype=np.uint8))
    writer.release()


def test_stage_local_review_preserves_source_frames_and_unreviewed_boundary(tmp_path: Path) -> None:
    source = tmp_path / "sample.mp4"
    _video(source)
    run_dir = tmp_path / "run"
    args = build_parser().parse_args([
        "run", "--input", str(source), "--output", str(run_dir),
        "--detector", "synthetic", "--tracker", "iou", "--manual-cut", "6",
    ])
    run_pipeline(args)

    output = tmp_path / "stage"
    completed = subprocess.run([
        sys.executable, "scripts/stage_local_review.py", "--source", str(source),
        "--run-dir", str(run_dir), "--output", str(output),
        "--manual-cut", "6", "--frames-per-shot", "2", "--skip-ocr",
    ], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

    summary = json.loads((output / "stage-summary.json").read_text())
    assert summary["status"] == "proposals_ready_review_required"
    assert summary["source_sha256"] == json.loads((run_dir / "run-manifest.json").read_text())["input_sha256"]
    assert summary["shot_ranges"] == {"shot-0": [0, 6], "shot-1": [6, 12]}
    assert summary["readiness_status"] == "not_ready"
    assert summary["identity_status"] == "not_attempted"

    review_pack = json.loads((output / "review-pack" / "review-pack.json").read_text())
    assert review_pack["reviewed"] is False
    assert {frame["source_frame"] for frame in review_pack["frames"]} == {0, 5, 6, 11}
    for shot_id, first_source_frame in (("shot-0", 0), ("shot-1", 6)):
        frame_map = json.loads((output / "cvat" / shot_id / "task-frame-map.json").read_text())
        assert frame_map["frames"][0]["source_frame"] == first_source_frame
        bundle = output / "cvat" / shot_id / "proposals.zip"
        with ZipFile(bundle) as archive:
            assert "annotations.xml" in archive.namelist()
    assert json.loads((output / "timing" / "shot-0.json").read_text())["reviewed"] is False


def test_stage_local_review_rejects_mismatched_source(tmp_path: Path) -> None:
    source = tmp_path / "sample.mp4"
    _video(source)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run-manifest.json").write_text(json.dumps({"input_sha256": "bad", "frame_count": 12}))
    completed = subprocess.run([
        sys.executable, "scripts/stage_local_review.py", "--source", str(source),
        "--run-dir", str(run_dir), "--output", str(tmp_path / "stage"), "--manual-cut", "6",
    ], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "source hash" in completed.stderr.lower()
    assert not (tmp_path / "stage" / "stage-summary.json").exists()


def test_stage_local_review_rejects_incomplete_inference(tmp_path: Path) -> None:
    source = tmp_path / "sample.mp4"
    _video(source)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run-manifest.json").write_text(json.dumps({
        "input_sha256": sha256_file(source), "frame_count": 12, "status": "failed",
    }))
    completed = subprocess.run([
        sys.executable, "scripts/stage_local_review.py", "--source", str(source),
        "--run-dir", str(run_dir), "--output", str(tmp_path / "stage"), "--manual-cut", "6",
    ], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "run manifest status" in completed.stderr.lower()
    assert not (tmp_path / "stage" / "stage-summary.json").exists()
