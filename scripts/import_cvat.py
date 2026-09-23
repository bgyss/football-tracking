#!/usr/bin/env python3
"""Import corrected CVAT video XML through its source-frame/PTS sidecar."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from football_tracking.annotations import load_annotation_manifest, mot_reference_from_manifest
from football_tracking.cvat import CvatError, TaskFrameMap, manifest_annotations_from_cvat, parse_cvat_video_xml, read_cvat_bundle
from football_tracking.metrics import sha256_file
from football_tracking.video import VideoInfo, frame_pts


def _shot(value: str) -> dict[str, Any]:
    parts = value.split(":")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("shot must be shot_id:start_frame:end_frame:camera_label:play_id:split")
    shot_id, start, end, camera, play, split = parts
    try:
        start_frame, end_frame = int(start), int(end)
    except ValueError as error:
        raise argparse.ArgumentTypeError("shot frame bounds must be integers") from error
    if not shot_id or not 0 <= start_frame < end_frame:
        raise argparse.ArgumentTypeError("shot needs an id and a valid non-empty frame interval")
    return {"shot_id": shot_id, "start_frame": start_frame, "end_frame": end_frame, "camera_label": camera or "unknown", "play_id": play or None, "split": split or "unassigned"}


def _all_records_reviewed(value: dict[str, Any]) -> bool:
    for collection_name in ("annotations", "landmarks", "frame_labels", "timing_events"):
        collection = value.get(collection_name, [])
        if not isinstance(collection, list):
            return False
        for record in collection:
            if not isinstance(record, dict) or record.get("review_status") not in {"reviewed", "accepted"}:
                return False
            if not all(record.get(key) not in (None, "") for key in ("reviewer", "revision", "reviewed_at", "annotation_confidence")):
                return False
    return True


def _stage_json(destination: Path, value: dict[str, Any]) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".import-tmp", dir=destination.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def import_cvat(
    cvat_path: Path,
    source_path: Path,
    output_path: Path,
    *,
    shot: dict[str, Any],
    frame_map_path: Path | None = None,
    template_path: Path | None = None,
    mark_reviewed: bool = False,
    reviewer: str | None = None,
    revision: int | None = None,
    reviewed_at: str | None = None,
    annotation_confidence: float | None = None,
    mot_reference_output: Path | None = None,
) -> None:
    source_hash = sha256_file(source_path)
    info = VideoInfo.from_path(source_path)
    xml_bytes, frame_map = read_cvat_bundle(cvat_path, frame_map_path, source_sha256=source_hash)
    if (frame_map.source_width, frame_map.source_height, frame_map.source_frame_count, frame_map.time_base) != (info.width, info.height, info.frame_count, info.time_base):
        raise CvatError("task frame map source metadata does not match the input video")
    pts_values = frame_pts(source_path)
    if len(pts_values) != info.frame_count:
        raise CvatError("unable to verify exact source PTS coverage for input video")
    for record in frame_map.frames:
        if pts_values[record.source_frame] != record.source_pts:
            raise CvatError(f"frame-map PTS mismatch at source frame {record.source_frame}")
    if not 0 <= shot["start_frame"] < shot["end_frame"] <= info.frame_count:
        raise CvatError("declared shot range lies outside source video")
    parsed_xml = parse_cvat_video_xml(xml_bytes)
    if parsed_xml.task_size is not None and parsed_xml.task_size != len(frame_map.frames):
        raise CvatError("CVAT task frame count does not match the source frame map; create the task with frame step 1")
    task_width, task_height = frame_map.frames[0].task_width, frame_map.frames[0].task_height
    if parsed_xml.task_width is not None and (parsed_xml.task_width, parsed_xml.task_height) != (task_width, task_height):
        raise CvatError("CVAT task dimensions do not match the source frame map crop transform")
    annotations, landmarks, frame_labels, point_proposals, timing_events = manifest_annotations_from_cvat(
        parsed_xml,
        frame_map,
        shot_id=shot["shot_id"],
        shot_range=(shot["start_frame"], shot["end_frame"]),
        reviewed=mark_reviewed,
        reviewer=reviewer,
        revision=revision,
        reviewed_at=reviewed_at,
        annotation_confidence=annotation_confidence,
    )
    if mark_reviewed and not annotations and not landmarks and not frame_labels and not timing_events:
        raise CvatError("cannot promote an empty CVAT import to reviewed")
    if mark_reviewed and point_proposals:
        raise CvatError(f"cannot promote import with {len(point_proposals)} unresolved point proposals; assign landmark semantics/roles or complete the point tracks first")
    if template_path is not None:
        value = json.loads(template_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise CvatError("annotation template must use schema_version 1")
        if value.get("source", {}).get("sha256") != source_hash:
            raise CvatError("annotation template source sha256 does not match input video")
        declared_shot = value.get("shots", {}).get(shot["shot_id"])
        if declared_shot is None:
            value.setdefault("shots", {})[shot["shot_id"]] = {key: shot[key] for key in ("start_frame", "end_frame", "play_id", "split", "camera_label")}
        elif (int(declared_shot["start_frame"]), int(declared_shot["end_frame"])) != (shot["start_frame"], shot["end_frame"]):
            raise CvatError("declared shot range does not match annotation template")
    else:
        value = {
            "schema_version": 1,
            "reviewed": False,
            "source": {"sha256": source_hash, "width": info.width, "height": info.height, "frame_count": info.frame_count, "time_base": list(info.time_base)},
            "shots": {shot["shot_id"]: {key: shot[key] for key in ("start_frame", "end_frame", "play_id", "split", "camera_label")}},
            "annotations": [],
            "landmarks": [],
            "timing_events": [],
            "frame_labels": [],
        }
    value.setdefault("annotations", []).extend(annotations)
    value.setdefault("landmarks", []).extend(landmarks)
    value.setdefault("timing_events", []).extend(timing_events)
    value.setdefault("frame_labels", []).extend(frame_labels)
    value.setdefault("point_proposals", []).extend(point_proposals)
    value["reviewed"] = bool(mark_reviewed and not value.get("point_proposals") and _all_records_reviewed(value))
    value["cvat_import"] = {
        "source_sha256": source_hash,
        "frame_map_schema_version": frame_map.schema_version,
        "imported_track_count": len(parsed_xml.tracks),
        "imported_shape_count": len(annotations),
        "imported_landmark_count": len(landmarks),
        "imported_frame_tag_count": len(frame_labels),
        "unresolved_point_proposal_count": len(point_proposals),
        "review_status": "reviewed" if value["reviewed"] else "unreviewed",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mot_reference_output is not None:
        if not value["reviewed"]:
            raise CvatError("MOT reference output requires a fully reviewed import")
        if mot_reference_output.resolve() == output_path.resolve():
            raise CvatError("MOT reference output must differ from the annotation manifest output")
    temporary_path = None
    reference_temporary_path = None
    manifest_backup_path = None
    try:
        temporary_path = _stage_json(output_path, value)
        if value["reviewed"]:
            parsed = load_annotation_manifest(temporary_path, source_hash, require_reviewed=True)
        else:
            parsed = load_annotation_manifest(temporary_path, source_hash, require_reviewed=False)
        if mot_reference_output is not None:
            reference = mot_reference_from_manifest(parsed)
            mot_reference_output.parent.mkdir(parents=True, exist_ok=True)
            reference_temporary_path = _stage_json(mot_reference_output, reference)
            if output_path.exists():
                descriptor, backup_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".import-backup", dir=output_path.parent)
                os.close(descriptor)
                manifest_backup_path = Path(backup_name)
                manifest_backup_path.unlink()
                output_path.replace(manifest_backup_path)
            try:
                temporary_path.replace(output_path)
                temporary_path = None
                reference_temporary_path.replace(mot_reference_output)
                reference_temporary_path = None
            except Exception:
                output_path.unlink(missing_ok=True)
                if manifest_backup_path is not None and manifest_backup_path.exists():
                    manifest_backup_path.replace(output_path)
                    manifest_backup_path = None
                raise
            if manifest_backup_path is not None:
                manifest_backup_path.unlink(missing_ok=True)
                manifest_backup_path = None
        else:
            temporary_path.replace(output_path)
            temporary_path = None
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if reference_temporary_path is not None:
            reference_temporary_path.unlink(missing_ok=True)
        if manifest_backup_path is not None and manifest_backup_path.exists() and not output_path.exists():
            manifest_backup_path.replace(output_path)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CVAT bundle .zip or annotations.xml")
    parser.add_argument("--frame-map", type=Path, help="required for standalone XML; bundles include this sidecar")
    parser.add_argument("--source", type=Path, required=True, help="original source video")
    parser.add_argument("--shot", type=_shot, required=True, help="shot_id:start_frame:end_frame:camera_label:play_id:split")
    parser.add_argument("--annotation-template", type=Path, help="existing manifest template to extend")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mot-reference-output", type=Path, help="optional MOT-style JSON reference output; requires --mark-reviewed")
    parser.add_argument("--mark-reviewed", action="store_true", help="promote this imported batch after full human review")
    parser.add_argument("--reviewer")
    parser.add_argument("--revision", type=int)
    parser.add_argument("--reviewed-at", help="ISO-8601 timestamp including timezone")
    parser.add_argument("--annotation-confidence", type=float)
    args = parser.parse_args()
    try:
        import_cvat(args.input, args.source, args.output, shot=args.shot, frame_map_path=args.frame_map, template_path=args.annotation_template, mark_reviewed=args.mark_reviewed, reviewer=args.reviewer, revision=args.revision, reviewed_at=args.reviewed_at, annotation_confidence=args.annotation_confidence, mot_reference_output=args.mot_reference_output)
    except (CvatError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
