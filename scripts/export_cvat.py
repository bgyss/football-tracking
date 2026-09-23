#!/usr/bin/env python3
"""Export source-addressed observation proposals to a CVAT video XML bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from football_tracking.cvat import CvatBox, CvatError, CvatPoint, CvatTrack, build_task_frame_map, source_bbox_to_task, source_point_to_task, write_cvat_bundle, write_cvat_video_xml


def _parse_crop(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        values = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("crop must be x1,y1,x2,y2") from error
    if len(values) != 4:
        raise argparse.ArgumentTypeError("crop must be x1,y1,x2,y2")
    return values  # type: ignore[return-value]


def _parse_size(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        width, height = (int(item) for item in value.lower().split("x", 1))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("task size must be WIDTHxHEIGHT") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("task size dimensions must be positive")
    return (width, height)


def export_observations(
    observations_path: Path,
    source_path: Path,
    *,
    shot_id: str,
    start_frame: int,
    end_frame: int,
    output_path: Path,
    cvat_xml_output: Path | None = None,
    crop_xyxy_px: tuple[float, float, float, float] | None = None,
    task_size: tuple[int, int] | None = None,
    review_pack_path: Path | None = None,
    frame_map_output: Path | None = None,
    max_gap_frames: int = 2,
    task_name: str = "football-tracking-proposals",
) -> None:
    if not shot_id:
        raise CvatError("shot_id must be non-empty")
    if not isinstance(max_gap_frames, int) or isinstance(max_gap_frames, bool) or max_gap_frames < 0:
        raise CvatError("max_gap_frames must be a non-negative integer")
    frame_map = build_task_frame_map(source_path, start_frame=start_frame, end_frame=end_frame, crop_xyxy_px=crop_xyxy_px, task_size=task_size)
    task_by_source = frame_map.by_source_frame()
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = {}
    with observations_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("shot_id", "")) != shot_id:
                continue
            try:
                source_frame = int(row["frame_index"])
                row_pts = int(row["pts"])
                bbox = tuple(float(item) for item in json.loads(row["bbox_xyxy_px"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise CvatError(f"invalid observation row for shot {shot_id}") from error
            if source_frame not in task_by_source:
                continue
            mapped = task_by_source[source_frame]
            if row_pts != mapped.source_pts:
                raise CvatError(f"observation PTS mismatch at source frame {source_frame}: {row_pts} != {mapped.source_pts}")
            tracklet_id = str(row.get("tracklet_id", "")).strip()
            if not tracklet_id:
                raise CvatError(f"observation at source frame {source_frame} has no tracklet_id")
            if len(bbox) != 4:
                raise CvatError(f"observation at source frame {source_frame} has invalid box")
            grouped.setdefault(tracklet_id, []).append((source_frame, row))

    tracks: list[CvatTrack] = []
    cvat_track_number = 0
    for tracklet_id, values in sorted(grouped.items()):
        values.sort(key=lambda item: item[0])
        segments: list[list[tuple[int, dict[str, str]]]] = []
        for frame, row in values:
            if not segments or frame - segments[-1][-1][0] > max_gap_frames + 1:
                segments.append([])
            segments[-1].append((frame, row))
        for segment_index, segment in enumerate(segments):
            boxes: list[CvatBox] = []
            for source_frame, row in segment:
                mapped = task_by_source[source_frame]
                try:
                    original_box = tuple(float(item) for item in json.loads(row["bbox_xyxy_px"]))
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise CvatError(f"invalid bbox for {tracklet_id} at {source_frame}") from error
                cx1, cy1, cx2, cy2 = mapped.crop_xyxy_px
                clipped = (max(original_box[0], cx1), max(original_box[1], cy1), min(original_box[2], cx2), min(original_box[3], cy2))
                if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                    continue
                coordinates = source_bbox_to_task(clipped, mapped)
                attrs = {
                    "source_frame": source_frame,
                    "source_pts": mapped.source_pts,
                    "detection_score": row.get("detection_score", ""),
                    "team_suggestion": row.get("team", "unknown"),
                    "team_score": row.get("team_score", "0"),
                    "ground_contact_proposal_xy_px": json.dumps([round((clipped[0] + clipped[2]) / 2.0, 3), round(clipped[3], 3)], separators=(",", ":")),
                    "review_status": "unreviewed",
                }
                boxes.append(CvatBox(mapped.task_frame, coordinates, keyframe=True, attributes=attrs))
            if not boxes:
                continue
            tracks.append(CvatTrack(
                cvat_track_number,
                "player",
                "auto",
                {
                    "tracklet_id": tracklet_id,
                    "proposal_segment": segment_index,
                    "source_sha256": frame_map.source_sha256,
                    "review_status": "unreviewed",
                },
                tuple(boxes),
            ))
            cvat_track_number += 1
            contact_shapes: list[CvatPoint] = []
            for source_frame, row in segment:
                mapped = task_by_source[source_frame]
                original_box = tuple(float(item) for item in json.loads(row["bbox_xyxy_px"]))
                contact = ((original_box[0] + original_box[2]) / 2.0, original_box[3])
                try:
                    contact_task_xy = source_point_to_task(contact, mapped)
                except CvatError:
                    continue
                contact_shapes.append(CvatPoint(mapped.task_frame, (contact_task_xy,), attributes={"source_frame": str(source_frame), "source_pts": str(mapped.source_pts), "method": "bbox_bottom_center", "review_status": "unreviewed"}))
            if contact_shapes:
                tracks.append(CvatTrack(
                    cvat_track_number,
                    "ground_contact",
                    "auto",
                    {"tracklet_id": tracklet_id, "proposal_method": "bbox_bottom_center", "source_sha256": frame_map.source_sha256, "review_status": "unreviewed"},
                    (),
                    tuple(contact_shapes),
                ))
                cvat_track_number += 1
    if review_pack_path is not None:
        try:
            review_pack = json.loads(review_pack_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CvatError(f"unable to read field proposal pack: {error}") from error
        if not isinstance(review_pack, dict) or review_pack.get("source", {}).get("sha256") != frame_map.source_sha256:
            raise CvatError("field proposal pack source sha256 does not match input video")
        mapped_frames = frame_map.by_source_frame()
        for record in review_pack.get("frames", []):
            if not isinstance(record, dict):
                continue
            source_frame = int(record.get("source_frame", -1))
            if source_frame not in mapped_frames:
                continue
            mapped = mapped_frames[source_frame]
            if int(record.get("pts", -1)) != mapped.source_pts:
                raise CvatError(f"field proposal PTS mismatch at source frame {source_frame}")
            for proposal_index, proposal in enumerate(record.get("field_intersection_proposals", [])):
                if not isinstance(proposal, dict) or not isinstance(proposal.get("point_xy_px"), list):
                    continue
                point_task_xy = source_point_to_task(proposal["point_xy_px"], mapped)
                tracks.append(CvatTrack(
                    cvat_track_number,
                    "field_landmark",
                    "auto",
                    {"proposal_id": f"{source_frame}:{proposal_index}", "landmark_id": "", "role": "", "source_sha256": frame_map.source_sha256, "review_status": "unreviewed"},
                    (),
                    (CvatPoint(mapped.task_frame, (point_task_xy,), attributes={"source_frame": str(source_frame), "source_pts": str(mapped.source_pts), "support_score": str(proposal.get("support_score", "")), "review_status": "unreviewed"}),),
                ))
                cvat_track_number += 1
    xml_bytes = write_cvat_video_xml(
        tracks,
        frame_map,
        task_name=task_name,
        label_attributes={
            "player": ("team", "global_id", "jersey_number", "visibility", "review_status", "ground_contact_xy_yards", "ground_contact_confidence", "cross_shot_review_status", "identity_second_reviewer", "identity_second_reviewed_at", "identity_second_revision", "identity_second_confidence"),
            "ground_contact": ("tracklet_id", "proposal_method", "source_sha256", "review_status", "ground_contact_confidence"),
            "field_landmark": ("proposal_id", "landmark_id", "role", "source_sha256", "review_status", "field_x_yards", "field_y_yards"),
        },
    )
    write_cvat_bundle(output_path, xml_bytes, frame_map)
    cvat_xml_path = cvat_xml_output or output_path.with_suffix(".xml")
    cvat_xml_path.parent.mkdir(parents=True, exist_ok=True)
    cvat_xml_path.write_bytes(xml_bytes)
    if frame_map_output is not None:
        frame_map_output.parent.mkdir(parents=True, exist_ok=True)
        frame_map_output.write_text(json.dumps(frame_map.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True, help="run observations.csv")
    parser.add_argument("--source", type=Path, required=True, help="original source video")
    parser.add_argument("--shot-id", required=True)
    parser.add_argument("--start-frame", type=int, required=True, help="inclusive source frame")
    parser.add_argument("--end-frame", type=int, required=True, help="exclusive source frame")
    parser.add_argument("--output", type=Path, required=True, help="CVAT bundle .zip")
    parser.add_argument("--cvat-xml-output", type=Path, help="standalone annotations.xml for import into an existing CVAT task; defaults beside the bundle")
    parser.add_argument("--crop", type=_parse_crop, help="source-space crop x1,y1,x2,y2 for a matching cropped CVAT task")
    parser.add_argument("--task-size", type=_parse_size, help="cropped task frame dimensions WIDTHxHEIGHT")
    parser.add_argument("--review-pack", type=Path, help="optional review-pack.json with unreviewed field-intersection proposals")
    parser.add_argument("--frame-map-output", type=Path, help="also write the source mapping as a standalone JSON sidecar for the later CVAT export")
    parser.add_argument("--max-gap-frames", type=int, default=2, help="split exported tracks after this many missing source frames")
    parser.add_argument("--task-name", default="football-tracking-proposals")
    args = parser.parse_args()
    try:
        export_observations(args.observations, args.source, shot_id=args.shot_id, start_frame=args.start_frame, end_frame=args.end_frame, output_path=args.output, cvat_xml_output=args.cvat_xml_output, crop_xyxy_px=args.crop, task_size=args.task_size, review_pack_path=args.review_pack, frame_map_output=args.frame_map_output, max_gap_frames=args.max_gap_frames, task_name=args.task_name)
    except (CvatError, OSError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
