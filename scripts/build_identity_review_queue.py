#!/usr/bin/env python3
"""Rank cross-shot identity cases for source-addressed human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from football_tracking.identity_review import IdentityReviewError, build_identity_review_queue, load_observation_index
from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo, frame_pts


def _draw_contact_sheet(queue: dict, source: Path, output: Path, limit: int) -> None:
    if limit < 1:
        raise ValueError("contact-sheet item limit must be positive")
    info = VideoInfo.from_path(source)
    pts_values = frame_pts(source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open source video {source}")
    cells: list[np.ndarray] = []
    target_size = (480, 270)
    try:
        for item in queue["items"][:limit]:
            for sample_index, sample in enumerate(item.get("samples", [])):
                row_cells = []
                for side_name, side in (("left", "A"), ("right", "B")):
                    record = sample.get(side_name)
                    cell = np.full((310, target_size[0], 3), 28, dtype=np.uint8)
                    if record is not None:
                        source_frame = int(record["source_frame"])
                        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError(f"unable to decode source frame {source_frame}")
                        reported = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                        if reported != source_frame:
                            raise RuntimeError(f"video seek returned frame {reported}, expected {source_frame}")
                        if source_frame >= len(pts_values) or int(record["source_pts"]) != pts_values[source_frame]:
                            raise RuntimeError(f"source PTS mismatch at frame {source_frame}")
                        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                        cell[40:310, :] = frame
                        label = f"{side}: {record['shot_id']} frame={source_frame} pts={record['source_pts']}"
                        cv2.putText(cell, label[:72], (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
                        x1, y1, x2, y2 = record["bbox_xyxy_px"]
                        sx, sy = target_size[0] / info.width, target_size[1] / info.height
                        cv2.rectangle(frame, (int(round(x1 * sx)), int(round(y1 * sy))), (int(round(x2 * sx)), int(round(y2 * sy))), (0, 220, 255), 2)
                        cell[40:310, :] = frame
                    else:
                        cv2.putText(cell, f"{side}: no mapped observation", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                    row_cells.append(cell)
                row = np.concatenate(row_cells, axis=1)
                header = np.full((42, row.shape[1], 3), 10, dtype=np.uint8)
                title = f"{item['review_item_id']} {item['kind']} priority={item['priority']} sample={sample_index + 1}"
                cv2.putText(header, title[:120], (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
                cells.append(np.concatenate((header, row), axis=0))
    finally:
        capture.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    if cells:
        canvas = np.concatenate(cells, axis=0)
    else:
        canvas = np.zeros((80, target_size[0] * 2, 3), dtype=np.uint8)
        cv2.putText(canvas, "No identity review items", (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"unable to write contact sheet {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-links", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="original video used to validate source frames and PTS")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--samples-per-pair", type=int, default=3)
    parser.add_argument("--contact-sheet", type=Path, help="optional original-frame side-by-side JPEG")
    parser.add_argument("--contact-sheet-items", type=int, default=24)
    args = parser.parse_args()
    try:
        source_hash = sha256_file(args.source)
        info = VideoInfo.from_path(args.source)
        pts_values = frame_pts(args.source)
        if len(pts_values) != info.frame_count:
            raise IdentityReviewError("exact source PTS coverage is incomplete")
        if any(right <= left for left, right in zip(pts_values, pts_values[1:])):
            raise IdentityReviewError("source PTS sequence must be strictly increasing")
        raw_links = json.loads(args.identity_links.read_text(encoding="utf-8"))
        observations = load_observation_index(args.observations)
        for tracklet_id, rows in observations.items():
            for row in rows:
                frame = int(row["source_frame"])
                if frame >= len(pts_values) or int(row["source_pts"]) != pts_values[frame]:
                    raise IdentityReviewError(f"observation PTS mismatch for {tracklet_id} at source frame {frame}")
                if tuple(row["time_base"]) != info.time_base:
                    raise IdentityReviewError(f"observation time_base mismatch for {tracklet_id} at source frame {frame}")
                x1, y1, x2, y2 = row["bbox_xyxy_px"]
                if x1 < 0 or y1 < 0 or x2 > info.width or y2 > info.height:
                    raise IdentityReviewError(f"observation box lies outside source frame for {tracklet_id} at source frame {frame}")
        queue = build_identity_review_queue(raw_links, observations, source_sha256=source_hash, max_items=args.max_items, samples_per_pair=args.samples_per_pair)
        queue["source"] = {"sha256": source_hash, "width": info.width, "height": info.height, "frame_count": info.frame_count, "time_base": list(info.time_base)}
        if args.contact_sheet is not None:
            _draw_contact_sheet(queue, args.source, args.contact_sheet, args.contact_sheet_items)
            queue["contact_sheet"] = str(args.contact_sheet)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError, IdentityReviewError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
