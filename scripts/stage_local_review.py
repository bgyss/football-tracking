#!/usr/bin/env python3
"""Run or reuse local tracking and stage source-addressed review proposals.

No proposal produced here is a reviewed calibration, play alignment, or identity.
The script makes no hosted API calls and never imports proposals as ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo


ROOT = Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> None:
    completed = subprocess.run([sys.executable, *arguments], cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"stage command failed ({completed.returncode}): {' '.join(arguments[:3])}")


def _sample_frames(start: int, end: int, count: int) -> list[int]:
    return sorted({start + round(index * (end - start - 1) / max(1, count - 1)) for index in range(count)})


def stage_review(
    source: Path,
    run_dir: Path,
    output: Path,
    *,
    manual_cuts: list[int],
    frames_per_shot: int = 4,
    skip_ocr: bool = False,
    ocr_samples_per_tracklet: int = 6,
    detector: str = "rfdetr",
    tracker: str = "botsort",
    detector_checkpoint: Path | None = None,
) -> dict:
    if frames_per_shot < 2 or ocr_samples_per_tracklet < 1:
        raise ValueError("frames per shot must be at least two and OCR samples must be positive")
    if not source.is_file():
        raise FileNotFoundError(source)
    info = VideoInfo.from_path(source)
    source_hash = sha256_file(source)
    cuts = sorted(set(manual_cuts))
    if any(cut <= 0 or cut >= info.frame_count for cut in cuts):
        raise ValueError("manual cuts must lie inside the source frame range")

    run_manifest_path = run_dir / "run-manifest.json"
    if not run_manifest_path.is_file():
        if detector == "rfdetr" and (detector_checkpoint is None or not detector_checkpoint.is_file()):
            raise ValueError("a local --detector-checkpoint is required for a new RF-DETR run")
        run_args = [
            "-m", "football_tracking", "run", "--input", str(source), "--output", str(run_dir),
            "--detector", detector, "--tracker", tracker,
        ]
        if detector_checkpoint is not None:
            run_args += ["--detector-checkpoint", str(detector_checkpoint)]
        for cut in cuts:
            run_args += ["--manual-cut", str(cut)]
        _run(*run_args)

    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("input_sha256") != source_hash:
        raise ValueError("run manifest source hash does not match source video")
    if manifest.get("frame_count") != info.frame_count:
        raise ValueError("run manifest frame count does not match source video")
    if manifest.get("status") not in {"complete", "complete_with_unresolved"}:
        raise ValueError("run manifest status is not complete")
    shots = json.loads((run_dir / "shots.json").read_text(encoding="utf-8"))
    ranges = [tuple(map(int, item)) for item in shots["ranges"]]
    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != info.frame_count or any(left[1] != right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError("run shot ranges do not cover the source exactly")
    if cuts and [item[0] for item in ranges[1:]] != cuts:
        raise ValueError("run shot boundaries do not match the requested manual cuts")
    for required in ("observations.csv", "analysis-config.json", "identity-links.json"):
        if not (run_dir / required).is_file():
            raise FileNotFoundError(run_dir / required)

    output.mkdir(parents=True, exist_ok=True)
    inventory_path = output / "play-inventory.json"
    inventory_args = ["scripts/build_play_inventory.py", "--input", str(source), "--output", str(inventory_path)]
    for _, end in ranges[:-1]:
        inventory_args += ["--manual-cut", str(end)]
    _run(*inventory_args)

    selected_frames = sorted({frame for start, end in ranges for frame in _sample_frames(start, end, frames_per_shot)})
    pack_dir = output / "review-pack"
    pack_args = ["scripts/build_identity_review_pack.py", "--input", str(source), "--output", str(pack_dir), "--observations", str(run_dir / "observations.csv")]
    for frame in selected_frames:
        pack_args += ["--frame", str(frame)]
    _run(*pack_args)

    shot_ranges: dict[str, list[int]] = {}
    for index, (start, end) in enumerate(ranges):
        shot_id = f"shot-{index}"
        shot_ranges[shot_id] = [start, end]
        _run(
            "scripts/propose_timing_events.py", "--source", str(source),
            "--start-frame", str(start), "--end-frame", str(end),
            "--output", str(output / "timing" / f"{shot_id}.json"),
        )
        cvat_dir = output / "cvat" / shot_id
        _run(
            "scripts/export_cvat.py", "--observations", str(run_dir / "observations.csv"),
            "--source", str(source), "--shot-id", shot_id,
            "--start-frame", str(start), "--end-frame", str(end),
            "--review-pack", str(pack_dir / "review-pack.json"),
            "--frame-map-output", str(cvat_dir / "task-frame-map.json"),
            "--output", str(cvat_dir / "proposals.zip"),
        )

    if not skip_ocr:
        _run(
            "scripts/propose_jersey_reads.py", "--source", str(source),
            "--observations", str(run_dir / "observations.csv"),
            "--analysis-config", str(run_dir / "analysis-config.json"),
            "--output", str(output / "jersey-cues.json"),
            "--max-samples-per-tracklet", str(ocr_samples_per_tracklet),
        )
    _run(
        "scripts/build_identity_review_queue.py", "--identity-links", str(run_dir / "identity-links.json"),
        "--observations", str(run_dir / "observations.csv"), "--source", str(source),
        "--output", str(output / "identity-review-queue.json"),
    )
    readiness_path = output / "readiness.json"
    _run(
        "scripts/check_identity_readiness.py", "--source", str(source),
        "--review-pack", str(pack_dir / "review-pack.json"), "--output", str(readiness_path),
    )
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    links = json.loads((run_dir / "identity-links.json").read_text(encoding="utf-8"))
    summary = {
        "schema_version": 1,
        "status": "proposals_ready_review_required",
        "source_sha256": source_hash,
        "run_id": manifest.get("run_id"),
        "detector": manifest.get("detector"),
        "tracker": manifest.get("tracker"),
        "shot_ranges": shot_ranges,
        "review_frames": selected_frames,
        "identity_status": links.get("status"),
        "readiness_status": readiness.get("status"),
        "ocr_status": "skipped" if skip_ocr else "unreviewed_proposals",
        "promotion_note": "Field intersections, timing bursts, OCR reads, and CVAT tracks remain unreviewed. No cross-shot identity is promoted by this command.",
    }
    (output / "stage-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manual-cut", type=int, action="append", default=[])
    parser.add_argument("--frames-per-shot", type=int, default=4)
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--ocr-samples-per-tracklet", type=int, default=6)
    parser.add_argument("--detector", choices=("rfdetr", "synthetic"), default="rfdetr")
    parser.add_argument("--tracker", choices=("botsort", "bytetrack", "iou"), default="botsort")
    parser.add_argument("--detector-checkpoint", type=Path)
    args = parser.parse_args()
    try:
        result = stage_review(
            args.source, args.run_dir, args.output, manual_cuts=args.manual_cut,
            frames_per_shot=args.frames_per_shot, skip_ocr=args.skip_ocr,
            ocr_samples_per_tracklet=args.ocr_samples_per_tracklet,
            detector=args.detector, tracker=args.tracker, detector_checkpoint=args.detector_checkpoint,
        )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"stage-local-review: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
