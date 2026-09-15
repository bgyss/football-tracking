"""Deterministic table, video, field-view, and timing exporters."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np

from .schema import Observation, RunManifest
from .video import ShotBoundary, VideoInfo, iter_video_frames


def _json_value(value: Any) -> str:
    return json.dumps(value, separators=(",", ": "))


def write_observations_csv(path: str | Path, observations: Iterable[Observation]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "shot_id",
        "frame_index",
        "pts",
        "time_base",
        "tracklet_id",
        "player_id",
        "play_id",
        "bbox_xyxy_px",
        "detection_score",
        "team",
        "team_score",
        "jersey_number",
        "field_xy_yards",
        "position_source",
        "calibration_id",
        "identity_version",
        "calibration_status",
        "calibration_reason",
        "position_uncertainty_yards",
        "play_time_s",
        "time_map_id",
        "source_tracklet_id",
    ]
    ordered = sorted(observations, key=lambda row: (row.frame_index, row.shot_id, row.tracklet_id))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for observation in ordered:
            value = observation.to_dict()
            value["bbox_xyxy_px"] = _json_value(value["bbox_xyxy_px"])
            value["time_base"] = _json_value(value["time_base"])
            value["field_xy_yards"] = _json_value(value["field_xy_yards"]) if value["field_xy_yards"] is not None else ""
            writer.writerow(value)


def write_observations_parquet(path: str | Path, observations: Iterable[Observation]) -> None:
    """Write the same observation contract in an analytics-friendly columnar format."""

    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("Parquet export requires the pyarrow dependency") from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [observation.to_dict() for observation in sorted(observations, key=lambda row: (row.frame_index, row.shot_id, row.tracklet_id))]
    table = pa.Table.from_pylist(rows)
    parquet.write_table(table, destination)


def write_trajectories_csv(path: str | Path, observations: Iterable[Observation]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["player_id", "shot_id", "frame_index", "pts", "x_px", "y_px", "x_yards", "y_yards", "position_source", "calibration_id", "calibration_status", "position_uncertainty_yards", "play_time_s", "time_map_id", "source_tracklet_id"]
    rows = [row for row in observations if row.player_id]
    rows.sort(key=lambda row: (row.player_id or "", row.frame_index, row.shot_id))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            x1, _, x2, y2 = row.bbox_xyxy_px
            x_yards, y_yards = row.field_xy_yards or (None, None)
            writer.writerow({"player_id": row.player_id, "shot_id": row.shot_id, "frame_index": row.frame_index, "pts": row.pts, "x_px": (x1 + x2) / 2.0, "y_px": y2, "x_yards": x_yards, "y_yards": y_yards, "position_source": row.position_source, "calibration_id": row.calibration_id, "calibration_status": row.calibration_status, "position_uncertainty_yards": row.position_uncertainty_yards, "play_time_s": row.play_time_s, "time_map_id": row.time_map_id, "source_tracklet_id": row.source_tracklet_id})


def write_calibration_json(path: str | Path, calibrations: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not calibrations:
        value: dict[str, Any] = {"schema_version": 1, "status": "not_provided", "shots": {}}
    else:
        statuses = {str(getattr(calibration, "status", "unvalidated")) for calibration in calibrations.values()}
        overall_status = "valid" if statuses == {"valid"} else "invalid" if statuses == {"invalid"} else "partial" if "partial" in statuses or "invalid" in statuses or len(statuses) > 1 else "unvalidated"
        value = {"schema_version": 1, "status": overall_status, "shots": {}}
        for shot_id, calibration in sorted(calibrations.items()):
            value["shots"][shot_id] = {
                "matrix": calibration.matrix,
                "median_error_px": calibration.median_error_px,
                "max_error_px": calibration.max_error_px,
                "fit_error_px": calibration.fit_error_px,
                "median_error_yards": calibration.median_error_yards,
                "max_error_yards": calibration.max_error_yards,
                "withheld_median_error_yards": calibration.withheld_median_error_yards,
                "withheld_p95_error_yards": calibration.withheld_p95_error_yards,
                "withheld_point_count": calibration.withheld_point_count,
                "inlier_count": calibration.inlier_count,
                "point_count": calibration.point_count,
                "reprojection_threshold_px": calibration.reprojection_threshold_px,
                "calibration_id": calibration.calibration_id,
                "status": calibration.status,
                "reason": calibration.reason,
            }
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_review_json(path: str | Path, review: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(review), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_identities_json(path: str | Path, identities: Mapping[str, str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(sorted(identities.items())), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manifest_json(path: str | Path, manifest: RunManifest | Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = manifest.to_dict() if isinstance(manifest, RunManifest) else dict(manifest)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class StageTimer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings[f"{name}_s"] = self.timings.get(f"{name}_s", 0.0) + (time.perf_counter() - started)


def _has_audio(path: Path) -> bool:
    command = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(completed.stdout.strip())


def _current_shot(frame_index: int, boundaries: Sequence[ShotBoundary]) -> str:
    starts = sorted(boundaries, key=lambda boundary: boundary.frame_index)
    current = 0
    for index, boundary in enumerate(starts):
        if boundary.frame_index <= frame_index:
            current = index
        else:
            break
    return f"shot-{current}"


def render_annotated_video(
    input_path: str | Path,
    output_path: str | Path,
    observations: Iterable[Observation],
    shots: Sequence[ShotBoundary],
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> None:
    """Render boxes, labels, and shot-local trails while retaining source audio when available."""

    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    info = VideoInfo.from_path(source)
    video_only = destination.with_name(destination.stem + ".video-only.mp4")
    writer = cv2.VideoWriter(str(video_only), cv2.VideoWriter_fourcc(*"mp4v"), info.source_fps, (info.width, info.height))
    if not writer.isOpened():
        raise RuntimeError(f"unable to create annotated video: {video_only}")
    by_frame: dict[int, list[Observation]] = {}
    for observation in observations:
        by_frame.setdefault(observation.frame_index, []).append(observation)
    traces: dict[tuple[str, str], list[tuple[int, int]]] = {}
    previous_shot: str | None = None
    if start_frame < 0 or (end_frame is not None and end_frame <= start_frame):
        raise ValueError("invalid annotated-video frame window")
    try:
        for frame_index, _, frame in iter_video_frames(source, start_frame=start_frame, end_frame=end_frame):
            shot_id = _current_shot(frame_index, shots)
            if previous_shot is not None and shot_id != previous_shot:
                traces.clear()
            previous_shot = shot_id
            for observation in sorted(by_frame.get(frame_index, ()), key=lambda row: row.tracklet_id):
                x1, y1, x2, y2 = (int(round(value)) for value in observation.bbox_xyxy_px)
                color = (255, 80, 40) if observation.team == "DET" else (40, 220, 255) if observation.team == "LAR" else (210, 210, 210)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = observation.player_id or observation.tracklet_id
                cv2.putText(frame, f"{label} {observation.team}", (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                contact = (int(round((x1 + x2) / 2)), y2)
                key = (observation.shot_id, observation.player_id or observation.tracklet_id)
                traces.setdefault(key, []).append(contact)
            for points in traces.values():
                if len(points) > 1:
                    cv2.polylines(frame, [np.asarray(points[-90:], dtype=np.int32)], False, (255, 180, 40), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    # A bounded window has no corresponding audio offset in the current
    # exporter; retain source audio only for a full-source render.
    if _has_audio(source) and start_frame == 0 and (end_frame is None or end_frame >= info.frame_count):
        muxed = destination.with_name(destination.stem + ".muxed.mp4")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_only), "-i", str(source), "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", str(muxed)]
        try:
            subprocess.run(command, check=True)
            muxed.replace(destination)
            video_only.unlink(missing_ok=True)
        except (OSError, subprocess.CalledProcessError):
            video_only.replace(destination)
    else:
        video_only.replace(destination)


def write_field_view(path: str | Path, trajectories: Mapping[tuple[str, str], Sequence[Sequence[float]]]) -> None:
    """Render field-space trails, one polyline per (player_id, shot_id).

    A player observed in two shots (e.g. a correct replay merge) draws two
    separate polylines sharing one identity label, rather than one continuous
    line joined by a spurious segment between shots — a play must never be
    double-counted in the field view.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 560
    margin_x, margin_y = 70, 60
    field_width, field_height = width - margin_x * 2, height - margin_y * 2
    canvas = np.full((height, width, 3), (35, 105, 35), dtype=np.uint8)
    cv2.rectangle(canvas, (margin_x, margin_y), (margin_x + field_width, margin_y + field_height), (240, 240, 240), 3)
    for yard in range(0, 121, 10):
        x = margin_x + int(field_width * yard / 120.0)
        cv2.line(canvas, (x, margin_y), (x, margin_y + field_height), (180, 220, 180), 1)
        cv2.putText(canvas, str(yard), (x - 10, margin_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    for (player_id, shot_id), points in sorted(trajectories.items()):
        normalized = [(float(point[0]), float(point[1])) for point in points if len(point) >= 2 and np.all(np.isfinite(point[:2]))]
        if not normalized:
            continue
        pixels = np.asarray([
            (margin_x + int(field_width * x / 120.0), margin_y + field_height - int(field_height * y / (160.0 / 3.0)))
            for x, y in normalized
        ], dtype=np.int32)
        color = tuple(int(value) for value in ((60, 180, 255) if player_id.endswith("1") else (255, 190, 60)))
        if len(pixels) > 1:
            cv2.polylines(canvas, [pixels], False, color, 3, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pixels[-1]), 5, color, -1)
        cv2.putText(canvas, player_id, tuple(pixels[-1] + np.array([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError(f"unable to write field view: {destination}")


def write_metrics_json(path: str | Path, metrics: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
