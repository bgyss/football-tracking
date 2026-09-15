#!/usr/bin/env python3
"""Extract exact source frames and unreviewed track proposals for human review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, action="append", required=True, help="exact source frame; repeat")
    parser.add_argument("--observations", type=Path, default=None, help="optional observations.csv for crop proposals")
    parser.add_argument("--exact-pts", action="store_true", help="scan and preserve the complete source PTS index; slower for long recordings")
    args = parser.parse_args()
    info = VideoInfo.from_path(args.input)
    pts_values: tuple[int, ...] = ()
    if args.exact_pts:
        from football_tracking.video import frame_pts

        pts_values = frame_pts(args.input)
    frames = sorted(set(args.frame))
    if any(frame < 0 or frame >= info.frame_count for frame in frames):
        raise SystemExit("requested frame lies outside the source video")
    proposals: dict[int, list[dict[str, object]]] = {frame: [] for frame in frames}
    if args.observations is not None:
        with args.observations.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                frame = int(row["frame_index"])
                if frame in proposals:
                    proposals[frame].append({"tracklet_id": row["tracklet_id"], "bbox_xyxy_px": json.loads(row["bbox_xyxy_px"])})
    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise SystemExit(f"unable to open {args.input}")
    records = []
    try:
        for frame in frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                raise SystemExit(f"unable to decode source frame {frame}")
            filename = f"frame-{frame:08d}.jpg"
            if not cv2.imwrite(str(frame_dir / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise SystemExit(f"unable to write {filename}")
            pts = pts_values[frame] if frame < len(pts_values) else int(round(frame * info.time_base[1] / (info.source_fps * info.time_base[0])))
            records.append({"source_frame": frame, "pts": pts, "pts_method": "source_index" if frame < len(pts_values) else "constant_rate_derivation", "image": f"frames/{filename}", "proposals": proposals[frame], "review_status": "unreviewed"})
    finally:
        capture.release()
    manifest = {
        "schema_version": 1,
        "reviewed": False,
        "source": {"path": str(args.input), "sha256": sha256_file(args.input), "width": info.width, "height": info.height, "frame_count": info.frame_count, "time_base": list(info.time_base)},
        "frames": records,
        "annotations": [],
        "review_policy": "Confirm exact source frame and PTS, convert all boxes to source coordinates, and review before evaluation.",
    }
    (args.output / "review-pack.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
