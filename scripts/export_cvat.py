#!/usr/bin/env python3
"""Export observations from one run directory as CVAT video preannotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from football_tracking.cvat import CvatBridgeError, build_cvat_preannotations
from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo, frame_pts


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CvatBridgeError(f"unable to read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise CvatBridgeError(f"{path.name} must contain a JSON object")
    return value


def _shot_metadata(value: dict, frame_count: int, play_id: str | None, pts_by_frame: Sequence[int]) -> dict[str, dict[str, object]]:
    ranges = value.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise CvatBridgeError("shots.json must contain non-empty source-frame ranges")
    raw_boundaries = value.get("boundaries", [])
    boundaries = {
        int(item["frame_index"]): int(item["pts"])
        for item in raw_boundaries
        if isinstance(item, dict) and "frame_index" in item and "pts" in item
    } if isinstance(raw_boundaries, list) else {}
    result: dict[str, dict[str, object]] = {}
    for index, interval in enumerate(ranges):
        if not isinstance(interval, list) or len(interval) != 2:
            raise CvatBridgeError(f"shots.json range {index} must be [start_frame, end_frame]")
        try:
            start, end = int(interval[0]), int(interval[1])
        except (TypeError, ValueError) as error:
            raise CvatBridgeError(f"shots.json range {index} must contain integers") from error
        if start < 0 or end <= start or end > frame_count:
            raise CvatBridgeError(f"shots.json range {index} lies outside the source")
        result[f"shot-{index}"] = {
            "start_frame": start,
            "end_frame": end,
            "play_id": play_id,
            "split": "unassigned",
            "camera_label": "unknown",
            "start_pts": pts_by_frame[start] if start < len(pts_by_frame) else boundaries.get(start),
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="directory containing observations.csv and run artifacts")
    parser.add_argument("--source", type=Path, required=True, help="original source video used by the run")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for annotations.xml and provenance.json")
    args = parser.parse_args(argv)

    try:
        run_dir = args.run_dir
        observations_path = run_dir / "observations.csv"
        run = _read_json(run_dir / "run-manifest.json")
        shots_value = _read_json(run_dir / "shots.json")
        config_path = run_dir / "analysis-config.json"
        config = _read_json(config_path) if config_path.exists() else {}
        source_hash = sha256_file(args.source)
        info = VideoInfo.from_path(args.source)
        pts_values = frame_pts(args.source)
        if str(run.get("input_sha256", "")) != source_hash:
            raise CvatBridgeError("run manifest input_sha256 does not match --source")
        if int(run.get("frame_count", -1)) != info.frame_count:
            raise CvatBridgeError("run manifest frame_count does not match --source")
        analysis = config.get("analysis", {}) if isinstance(config.get("analysis", {}), dict) else {}
        play_id_value = analysis.get("play_id")
        shots = _shot_metadata(shots_value, info.frame_count, None if play_id_value is None else str(play_id_value), pts_values)
        source = {
            "sha256": source_hash,
            "width": info.width,
            "height": info.height,
            "frame_count": info.frame_count,
            "time_base": list(info.time_base),
        }
        with observations_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        xml_text, provenance = build_cvat_preannotations(
            rows,
            source=source,
            run=run,
            shots=shots,
            observations_csv_sha256=sha256_file(observations_path),
            pts_by_frame=pts_values,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "annotations.xml").write_text(xml_text + "\n", encoding="utf-8")
        (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, CvatBridgeError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
