#!/usr/bin/env python3
"""Extract exact source frames and unreviewed track proposals for human review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo, frame_pts


def field_line_proposals(image_bgr) -> list[dict[str, object]]:
    """Return unreviewed white-line segment hints for calibration review."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    lines = cv2.HoughLinesP(edges, 1.0, 3.141592653589793 / 180.0, threshold=45, minLineLength=35, maxLineGap=12)
    if lines is None:
        return []
    candidates: list[tuple[float, list[int]]] = []
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = (int(value) for value in line)
        length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        candidates.append((length, [x1, y1, x2, y2]))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [{"segment_xyxy_px": segment, "length_px": length, "review_status": "unreviewed"} for length, segment in candidates[:100]]


def field_intersection_proposals(image_bgr, line_proposals: list[dict[str, object]]) -> list[dict[str, object]]:
    """Propose image intersections of line segments without assigning field semantics."""

    height, width = image_bgr.shape[:2]
    lines = [item["segment_xyxy_px"] for item in line_proposals]
    candidates: list[tuple[float, float, float, dict[str, object]]] = []
    for first_index, first in enumerate(lines):
        ax1, ay1, ax2, ay2 = (float(value) for value in first)
        adx, ady = ax2 - ax1, ay2 - ay1
        alength = (adx * adx + ady * ady) ** 0.5
        for second_index in range(first_index + 1, len(lines)):
            second = lines[second_index]
            bx1, by1, bx2, by2 = (float(value) for value in second)
            bdx, bdy = bx2 - bx1, by2 - by1
            blength = (bdx * bdx + bdy * bdy) ** 0.5
            cross = adx * bdy - ady * bdx
            if alength < 1.0 or blength < 1.0 or abs(cross) < 0.25 * alength * blength:
                continue
            dx, dy = bx1 - ax1, by1 - ay1
            t = (dx * bdy - dy * bdx) / cross
            u = (dx * ady - dy * adx) / cross
            if not -0.15 <= t <= 1.15 or not -0.15 <= u <= 1.15:
                continue
            x, y = ax1 + t * adx, ay1 + t * ady
            if not 4.0 <= x < width - 4.0 or not 4.0 <= y < height - 4.0:
                continue
            orthogonality = abs(cross) / (alength * blength)
            support = min(alength, blength) * (0.5 + 0.5 * orthogonality)
            record = {
                "point_xy_px": [round(x, 3), round(y, 3)],
                "support_line_indices": [first_index, second_index],
                "support_score": round(support, 3),
                "semantic_landmark_id": None,
                "review_status": "unreviewed",
            }
            candidates.append((support, x, y, record))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected: list[dict[str, object]] = []
    for _, x, y, record in candidates:
        if any((float(item["point_xy_px"][0]) - x) ** 2 + (float(item["point_xy_px"][1]) - y) ** 2 < 144.0 for item in selected):
            continue
        selected.append(record)
        if len(selected) >= 200:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, action="append", required=True, help="exact source frame; repeat")
    parser.add_argument("--observations", type=Path, default=None, help="optional observations.csv for crop proposals")
    parser.add_argument("--exact-pts", action="store_true", help="deprecated; source PTS indexing is always exact")
    args = parser.parse_args()
    info = VideoInfo.from_path(args.input)
    pts_values = frame_pts(args.input)
    if len(pts_values) != info.frame_count:
        raise SystemExit(f"exact source PTS coverage is incomplete: {len(pts_values)} values for {info.frame_count} frames")
    if any(right <= left for left, right in zip(pts_values, pts_values[1:])):
        raise SystemExit("source PTS sequence must be strictly increasing")
    frames = sorted(set(args.frame))
    if any(frame < 0 or frame >= info.frame_count for frame in frames):
        raise SystemExit("requested frame lies outside the source video")
    proposals: dict[int, list[dict[str, object]]] = {frame: [] for frame in frames}
    if args.observations is not None:
        with args.observations.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                frame = int(row["frame_index"])
                if frame in proposals:
                    bbox = json.loads(row["bbox_xyxy_px"])
                    proposals[frame].append({
                        "tracklet_id": row["tracklet_id"],
                        "bbox_xyxy_px": bbox,
                        "detection_score": float(row.get("detection_score") or 0.0),
                        "team_suggestion": row.get("team") or "unknown",
                        "team_score": float(row.get("team_score") or 0.0),
                        "ground_contact_proposal": {
                            "point_xy_px": [round((float(bbox[0]) + float(bbox[2])) / 2.0, 3), float(bbox[3])],
                            "method": "bbox_bottom_center",
                            "ground_contact_confidence": None,
                            "review_status": "unreviewed",
                        },
                        "review_status": "unreviewed",
                    })
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
            reported_frame = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if reported_frame != frame:
                raise SystemExit(f"video seek returned frame {reported_frame}, expected {frame}")
            filename = f"frame-{frame:08d}.jpg"
            if not cv2.imwrite(str(frame_dir / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise SystemExit(f"unable to write {filename}")
            lines = field_line_proposals(image)
            records.append({"source_frame": frame, "pts": pts_values[frame], "pts_method": "source_index", "image": f"frames/{filename}", "proposals": proposals[frame], "field_line_proposals": lines, "field_intersection_proposals": field_intersection_proposals(image, lines), "review_status": "unreviewed"})
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
