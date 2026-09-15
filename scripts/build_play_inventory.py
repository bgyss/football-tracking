#!/usr/bin/env python3
"""Scan a game into unreviewed shot-interval candidates for play annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_tracking.cli import _boundaries_for_video
from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo, shot_ranges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manual-cut", type=int, action="append", default=[])
    parser.add_argument("--exact-pts", action="store_true", help="scan and preserve the complete source PTS index")
    parser.add_argument("--scene-threshold", type=float, default=0.08, help="unreviewed scene-change threshold for candidate cuts")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"input video does not exist: {args.input}")
    info = VideoInfo.from_path(args.input)
    pts_values: tuple[int, ...] = ()
    if args.exact_pts:
        from football_tracking.video import frame_pts

        pts_values = frame_pts(args.input)
    boundaries = _boundaries_for_video(info, args.manual_cut, scene_threshold=args.scene_threshold)
    intervals = shot_ranges(info.frame_count, boundaries)
    shots = []
    for index, (start, end) in enumerate(intervals):
        start_pts = pts_values[start] if start < len(pts_values) else int(round(start * info.time_base[1] / (info.source_fps * info.time_base[0])))
        if end < len(pts_values):
            end_pts = pts_values[end]
        elif pts_values:
            step = pts_values[-1] - pts_values[-2] if len(pts_values) > 1 else int(round(info.time_base[1] / (info.source_fps * info.time_base[0])))
            end_pts = pts_values[-1] + max(1, step)
        else:
            end_pts = int(round(end * info.time_base[1] / (info.source_fps * info.time_base[0])))
        boundary = next((item for item in boundaries if item.frame_index == start), None)
        shots.append({"shot_id": f"shot-{index}", "start_frame": start, "end_frame": end, "start_pts": start_pts, "end_pts": end_pts, "pts_method": "source_index" if pts_values else "constant_rate_derivation", "boundary_reason": boundary.reason if boundary else "derived", "boundary_confidence": boundary.confidence if boundary else 0.0, "camera_label": None, "play_id": None, "split": None, "review_status": "unreviewed"})
    artifact = {"schema_version": 1, "reviewed": False, "source": {"path": str(args.input), "sha256": sha256_file(args.input), "width": info.width, "height": info.height, "frame_count": info.frame_count, "source_fps": info.source_fps, "time_base": list(info.time_base)}, "shots": shots, "scouting_policy": {"scene_threshold": args.scene_threshold, "thumbnail_cap": 8000}, "review_policy": "Confirm every boundary, assign camera_label/play_id/split, and review before using for calibration or identity."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
