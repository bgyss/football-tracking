from __future__ import annotations

import cv2
import json
import numpy as np
import csv

from football_tracking.cli import build_parser, main
from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo, frame_pts


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


def test_parser_accepts_play_selection_for_multi_play_alignment() -> None:
    args = build_parser().parse_args(["run", "--input", "video.mp4", "--output", "out", "--play-alignment", "alignments.json", "--play-id", "p1"])
    assert args.play_id == "p1"


def test_batch_runs_each_reviewed_play_in_its_own_directory(tmp_path) -> None:
    input_path = tmp_path / "batch.mp4"
    make_video(input_path)
    alignment = tmp_path / "alignments.json"
    alignment.write_text(json.dumps({"reviewed": True, "plays": [
        {"play_id": "p1", "anchors": [{"shot_id": "shot-0", "source_frame": 0}, {"shot_id": "shot-1", "source_frame": 2}], "shot_ranges": {"shot-0": [0, 2], "shot-1": [2, 4]}},
        {"play_id": "p2", "anchors": [{"shot_id": "shot-0", "source_frame": 0}, {"shot_id": "shot-1", "source_frame": 2}], "shot_ranges": {"shot-0": [0, 2], "shot-1": [2, 4]}},
    ]}), encoding="utf-8")
    output = tmp_path / "batch-output"
    assert main(["batch", "--input", str(input_path), "--output", str(output), "--detector", "synthetic", "--tracker", "iou", "--play-alignment", str(alignment)]) == 0
    report = json.loads((output / "batch.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete_with_unresolved"
    assert [item["play_id"] for item in report["plays"]] == ["p1", "p2"]
    assert all(item["promotion_gate_status"] == "not_evaluated" for item in report["plays"])
    assert all((output / play / "identity-links.json").is_file() for play in ("p1", "p2"))


def test_batch_preserves_declared_shot_ids_when_window_starts_after_frame_zero(tmp_path) -> None:
    input_path = tmp_path / "later-batch.mp4"
    make_video(input_path)
    alignment = tmp_path / "later-alignments.json"
    alignment.write_text(json.dumps({"reviewed": True, "plays": [
        {"play_id": "p-later", "anchors": [{"shot_id": "shot-2", "source_frame": 2}, {"shot_id": "shot-3", "source_frame": 3}], "shot_ranges": {"shot-2": [2, 3], "shot-3": [3, 4]}},
    ]}), encoding="utf-8")
    output = tmp_path / "later-batch-output"
    assert main(["batch", "--input", str(input_path), "--output", str(output), "--detector", "synthetic", "--tracker", "iou", "--play-alignment", str(alignment)]) == 0
    with (output / "p-later" / "observations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["shot_id"] for row in rows} == {"shot-2", "shot-3"}


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


def test_run_rejects_calibration_source_hash_before_detection(tmp_path) -> None:
    input_path = tmp_path / "hash-mismatch.mp4"
    make_video(input_path)
    calibration = tmp_path / "bad-calibration.json"
    calibration.write_text(json.dumps({"schema_version": 2, "source_sha256": "wrong", "shots": {}}), encoding="utf-8")
    output = tmp_path / "mismatch-output"
    assert main(["run", "--input", str(input_path), "--output", str(output), "--detector", "synthetic", "--calibration", str(calibration)]) == 2
    assert not (output / "detections.jsonl").exists()


def test_run_rejects_reference_source_hash_before_detection(tmp_path) -> None:
    input_path = tmp_path / "reference-hash-mismatch.mp4"
    make_video(input_path)
    reference = tmp_path / "bad-reference.json"
    reference.write_text(json.dumps({"reviewed": True, "source_sha256": "wrong", "sequences": {"shot-0": {"frames": {}}}}), encoding="utf-8")
    output = tmp_path / "reference-mismatch-output"
    assert main(["run", "--input", str(input_path), "--output", str(output), "--detector", "synthetic", "--reviewed-reference", str(reference)]) == 2
    assert not (output / "detections.jsonl").exists()


def test_run_rejects_reviewed_split_source_hash_before_detection(tmp_path) -> None:
    input_path = tmp_path / "split-hash-mismatch.mp4"
    make_video(input_path)
    splits = tmp_path / "bad-splits.json"
    splits.write_text(json.dumps({"reviewed": True, "source_sha256": "wrong", "splits": {}}), encoding="utf-8")
    output = tmp_path / "split-mismatch-output"
    assert main(["run", "--input", str(input_path), "--output", str(output), "--detector", "synthetic", "--reviewed-splits", str(splits)]) == 2
    assert not (output / "detections.jsonl").exists()


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
    assert report["schema_version"] == 2
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
    # The legacy static file still exposes field coordinates, but it is not
    # eligible to create a cross-shot merge without withheld validation.
    assert report.get("unresolved_shots") == []
    assert report["status"] == "abstained"
    assert report["reason"].startswith("legacy static calibration")


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


def write_validated_alignment(path, input_path, frames_per_shot: int) -> None:
    pts = frame_pts(input_path)
    assert len(pts) >= frames_per_shot * 2
    values = {
        "reviewed": True,
        "source_sha256": sha256_file(input_path),
        "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 0, "source_pts": pts[0], "play_time_s": 0.0, "event": "snap"},
            {"shot_id": "shot-1", "source_frame": frames_per_shot, "source_pts": pts[frames_per_shot], "play_time_s": 0.0, "event": "snap"},
        ],
        "correspondences": [
            {"shot_id": "shot-0", "source_pts": pts[frames_per_shot - 1], "play_time_s": (frames_per_shot - 1) / 10.0, "event": "contact"},
            {"shot_id": "shot-1", "source_pts": pts[frames_per_shot * 2 - 1], "play_time_s": (frames_per_shot - 1) / 10.0, "event": "contact"},
        ],
        "validation_correspondences": [
            {"shot_id": "shot-0", "source_pts": pts[3], "play_time_s": 0.3, "event": "release"},
            {"shot_id": "shot-1", "source_pts": pts[frames_per_shot + 3], "play_time_s": 0.3, "event": "release"},
        ],
    }
    path.write_text(json.dumps(values), encoding="utf-8")


def test_run_reaches_a_resolved_cross_shot_outcome_from_view_invariant_evidence(tmp_path) -> None:
    frames_per_shot = 7
    input_path = tmp_path / "replay.mp4"
    make_replay_video(input_path, frames_per_shot)

    alignment = tmp_path / "alignment.json"
    write_validated_alignment(alignment, input_path, frames_per_shot)

    landmarks = {
        "image_points": [[0, 0], [REPLAY_WIDTH, 0], [REPLAY_WIDTH, REPLAY_HEIGHT], [0, REPLAY_HEIGHT]],
        "field_points": [[0, 0], [120, 0], [120, 53.33], [0, 53.33]],
        "withheld_image_points": [[REPLAY_WIDTH / 2, REPLAY_HEIGHT / 2]],
        "withheld_field_points": [[60, 26.665]],
        "pts_start": 0,
        "pts_end": 100000,
    }
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"schema_version": 2, "source_sha256": sha256_file(input_path), "shots": {"shot-0": {"keyframes": [landmarks]}, "shot-1": {"keyframes": [landmarks]}}}), encoding="utf-8")

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
    assert report["calibration_status"] == "timeline_withheld_validated"
    assert report["alignment_source_hash_validated"] is True
    assert report["cross_shot_player_ids"] > 0
    assert {link["decision"] for link in report["links"]} == {"same"}
    assert report["player_id_count"] < report["tracklet_count"]

    review = json.loads((output / "review.json").read_text(encoding="utf-8"))
    assert review["status"] == "resolved_cross_view"
    assert review["cross_view_identity_resolved"] is True

    manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"


def test_run_accepts_pts_scoped_calibration_timeline(tmp_path) -> None:
    frames_per_shot = 7
    input_path = tmp_path / "replay-timeline.mp4"
    make_replay_video(input_path, frames_per_shot)
    alignment = tmp_path / "alignment.json"
    write_validated_alignment(alignment, input_path, frames_per_shot)
    base = {
        "image_points": [[0, 0], [REPLAY_WIDTH, 0], [REPLAY_WIDTH, REPLAY_HEIGHT], [0, REPLAY_HEIGHT]],
        "field_points": [[0, 0], [120, 0], [120, 53.33], [0, 53.33]],
        "withheld_image_points": [[REPLAY_WIDTH / 2, REPLAY_HEIGHT / 2]],
        "withheld_field_points": [[60, 26.665]],
        "pts_start": 0,
        "pts_end": 100000,
    }
    calibration = tmp_path / "calibration-timeline.json"
    calibration.write_text(json.dumps({"schema_version": 2, "source_sha256": sha256_file(input_path), "shots": {"shot-0": {"keyframes": [base]}, "shot-1": {"keyframes": [base]}}}), encoding="utf-8")
    prototypes = tmp_path / "prototypes.json"
    prototypes.write_text(json.dumps({"blue": [0.0, 0.0, 1.0], "green": [0.0, 1.0, 0.0]}), encoding="utf-8")
    output = tmp_path / "timeline-run"
    assert main(["run", "--input", str(input_path), "--output", str(output), "--detector", "synthetic", "--tracker", "iou", "--manual-cut", str(frames_per_shot), "--play-alignment", str(alignment), "--calibration", str(calibration), "--team-prototypes", str(prototypes)]) == 0
    report = json.loads((output / "identity-links.json").read_text(encoding="utf-8"))
    assert report["status"] == "resolved"
    exported = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    assert exported["schema_version"] == 2


def test_analysis_identity_includes_calibration_provenance(tmp_path) -> None:
    input_path = tmp_path / "hash.mp4"
    make_video(input_path)
    first_calibration = tmp_path / "first.json"
    second_calibration = tmp_path / "second.json"
    base = {"image_points": [[0, 0], [32, 0], [32, 24], [0, 24]], "field_points": [[0, 0], [120, 0], [120, 53.33], [0, 53.33]]}
    first_calibration.write_text(json.dumps(base), encoding="utf-8")
    second_calibration.write_text(json.dumps({**base, "reprojection_threshold_px": 4.0}), encoding="utf-8")
    first_output = tmp_path / "first-run"
    second_output = tmp_path / "second-run"
    assert main(["run", "--input", str(input_path), "--output", str(first_output), "--detector", "synthetic", "--tracker", "iou", "--calibration", str(first_calibration)]) == 0
    assert main(["run", "--input", str(input_path), "--output", str(second_output), "--detector", "synthetic", "--tracker", "iou", "--calibration", str(second_calibration)]) == 0
    first = json.loads((first_output / "run-manifest.json").read_text(encoding="utf-8"))
    second = json.loads((second_output / "run-manifest.json").read_text(encoding="utf-8"))
    assert first["config_hash"] != second["config_hash"]
    config = json.loads((first_output / "analysis-config.json").read_text(encoding="utf-8"))
    assert config["schema_version"] == 2
    assert config["analysis_hash"] == first["config_hash"]
    assert config["source_sha256"] == first["input_sha256"]
    assert config["analysis"]["observation_schema_version"] == 2
    assert "evaluator_version" in config["analysis"]


def test_run_supports_bounded_source_windows_and_records_scope(tmp_path) -> None:
    input_path = tmp_path / "window.mp4"
    make_video(input_path)
    output = tmp_path / "window-run"
    assert main(["run", "--input", str(input_path), "--output", str(output), "--detector", "synthetic", "--tracker", "iou", "--start-frame", "1", "--end-frame", "3"]) == 0
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["window"] == {"start_frame": 1, "end_frame": 3, "processed_frame_count": 2, "source_frame_count": 4}
    with (output / "observations.csv").open(newline="") as handle:
        assert {int(row["frame_index"]) for row in csv.DictReader(handle)} == {1, 2}
    capture = cv2.VideoCapture(str(output / "annotated.mp4"))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    assert count == 2
    assert main(["run", "--input", str(input_path), "--output", str(output), "--detector", "synthetic", "--tracker", "iou", "--start-frame", "0", "--end-frame", "4"]) == 0
    rerun_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert rerun_metrics["detection_cache_hit"] is False


def test_run_promotes_only_a_complete_reviewed_synthetic_fixture(tmp_path) -> None:
    import pytest

    pytest.importorskip("trackeval")
    frames_per_shot = 7
    input_path = tmp_path / "promoted.mp4"
    make_replay_video(input_path, frames_per_shot)
    alignment = tmp_path / "alignment.json"
    write_validated_alignment(alignment, input_path, frames_per_shot)
    calibration_config = {
        "image_points": [[0, 0], [REPLAY_WIDTH, 0], [REPLAY_WIDTH, REPLAY_HEIGHT], [0, REPLAY_HEIGHT]],
        "field_points": [[0, 0], [120, 0], [120, 53.33], [0, 53.33]],
        "withheld_image_points": [[REPLAY_WIDTH / 2, REPLAY_HEIGHT / 2]],
        "withheld_field_points": [[60, 26.665]],
        "pts_start": 0,
        "pts_end": 100000,
    }
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"schema_version": 2, "source_sha256": sha256_file(input_path), "shots": {"shot-0": {"keyframes": [calibration_config]}, "shot-1": {"keyframes": [calibration_config]}}}), encoding="utf-8")
    prototypes = tmp_path / "prototypes.json"
    prototypes.write_text(json.dumps({"blue": [0.0, 0.0, 1.0], "green": [0.0, 1.0, 0.0]}), encoding="utf-8")
    boxes = [[10.0, 12.0, 18.0, 46.8], [62.0, 10.8, 70.0, 45.6]]
    sequences = {}
    for shot_id, start in (("shot-0", 0), ("shot-1", frames_per_shot)):
        frames = {}
        for index in range(frames_per_shot):
            frames[str(start + index)] = {"labeled": True, "objects": [
                {"id": "p1" if shot_id == "shot-0" else "q1", "bbox_xyxy": boxes[0], "ground_contact_xy_yards": [16.8, 41.6], "team": "blue"},
                {"id": "p2" if shot_id == "shot-0" else "q2", "bbox_xyxy": boxes[1], "ground_contact_xy_yards": [79.2, 40.5], "team": "green"},
            ]}
        sequences[shot_id] = {"frames": frames}
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"reviewed": True, "source_sha256": sha256_file(input_path), "sequences": sequences, "cross_shot_identity": {"shot-0": {"p1": "A", "p2": "B"}, "shot-1": {"q1": "A", "q2": "B"}}}), encoding="utf-8")
    output = tmp_path / "promoted-run"
    assert main(["run", "--input", str(input_path), "--output", str(output), "--detector", "synthetic", "--tracker", "iou", "--manual-cut", str(frames_per_shot), "--play-alignment", str(alignment), "--calibration", str(calibration), "--team-prototypes", str(prototypes), "--reviewed-reference", str(reference)]) == 0
    evaluation = json.loads((output / "tracking-evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["promotion_gate"]["status"] == "not_evaluated"
    assert "proxy detector" in evaluation["promotion_gate"]["reasons"][0]
    assert evaluation["cross_shot"]["false_merges"] == 0
    assert evaluation["ground_contact"]["gate"]["valid_fraction_at_least_0_90"] is True
    artifact_validation = json.loads((output / "artifact-validation.json").read_text(encoding="utf-8"))
    assert artifact_validation["status"] == "valid"
