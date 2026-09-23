#!/usr/bin/env python3
"""Scan a game into unreviewed shot-interval candidates for play annotation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

from football_tracking.cli import _boundaries_for_video
from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo, frame_pts, shot_ranges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manual-cut", type=int, action="append", default=[])
    parser.add_argument("--exact-pts", action="store_true", help="deprecated; source PTS indexing is always exact")
    parser.add_argument("--scene-threshold", type=float, default=0.08, help="unreviewed scene-change threshold for candidate cuts")
    parser.add_argument("--candidate-window-shots", type=int, default=4, help="maximum adjacent shots in each unreviewed play-window proposal")
    parser.add_argument("--candidate-window-max-s", type=float, default=35.0, help="maximum duration of an unreviewed play-window proposal")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"input video does not exist: {args.input}")
    info = VideoInfo.from_path(args.input)
    if args.candidate_window_shots < 2 or not math.isfinite(args.candidate_window_max_s) or args.candidate_window_max_s <= 0:
        raise SystemExit("candidate play-window limits must be positive (at least two shots)")
    pts_values = frame_pts(args.input)
    if len(pts_values) != info.frame_count:
        raise SystemExit(f"exact source PTS coverage is incomplete: {len(pts_values)} values for {info.frame_count} frames")
    if any(right <= left for left, right in zip(pts_values, pts_values[1:])):
        raise SystemExit("source PTS sequence must be strictly increasing")
    boundaries = [replace(item, pts=pts_values[item.frame_index]) for item in _boundaries_for_video(info, args.manual_cut, scene_threshold=args.scene_threshold)]
    intervals = shot_ranges(info.frame_count, boundaries)
    shots = []
    for index, (start, end) in enumerate(intervals):
        start_pts = pts_values[start]
        end_pts = pts_values[end] if end < len(pts_values) else None
        boundary = next((item for item in boundaries if item.frame_index == start), None)
        shots.append({"shot_id": f"shot-{index}", "start_frame": start, "end_frame": end, "start_pts": start_pts, "end_pts": end_pts, "start_pts_method": "source_index", "end_pts_method": "source_index" if end_pts is not None else "exclusive_eof_no_source_pts", "boundary_reason": boundary.reason if boundary else "derived", "boundary_confidence": boundary.confidence if boundary else 0.0, "camera_label": None, "play_id": None, "split": None, "review_status": "unreviewed"})
    play_windows = []
    for start_index in range(len(shots)):
        for shot_count in range(2, min(args.candidate_window_shots, len(shots) - start_index) + 1):
            first, last = shots[start_index], shots[start_index + shot_count - 1]
            duration_s = (last["end_pts"] - first["start_pts"]) * info.time_base[0] / info.time_base[1] if last["end_pts"] is not None else (last["end_frame"] - first["start_frame"]) / info.source_fps
            if duration_s > args.candidate_window_max_s:
                break
            play_windows.append({
                "candidate_id": f"play-candidate-{len(play_windows):06d}",
                "shot_ids": [shot["shot_id"] for shot in shots[start_index : start_index + shot_count]],
                "start_frame": first["start_frame"],
                "end_frame": last["end_frame"],
                "start_pts": first["start_pts"],
                "end_pts_exclusive": last["end_pts"],
                "duration_s": round(float(duration_s), 6),
                "duration_method": "source_pts" if last["end_pts"] is not None else "frame_rate_to_eof_estimate",
                "review_status": "unreviewed",
                "proposal_reason": "adjacent_short_shot_window; confirm whether views show the same play",
            })
    artifact = {"schema_version": 1, "reviewed": False, "source": {"path": str(args.input), "sha256": sha256_file(args.input), "width": info.width, "height": info.height, "frame_count": info.frame_count, "source_fps": info.source_fps, "time_base": list(info.time_base)}, "shots": shots, "candidate_play_windows": play_windows, "scouting_policy": {"scene_threshold": args.scene_threshold, "thumbnail_cap": 8000, "candidate_window_shots": args.candidate_window_shots, "candidate_window_max_s": args.candidate_window_max_s}, "review_policy": "Confirm every boundary, camera_label, play grouping, and split before using for calibration or identity."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
