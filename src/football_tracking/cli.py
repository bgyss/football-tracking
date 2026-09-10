"""Command-line entry points for local football tracking and benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .cache import CacheMismatch, DetectionCache
from .calibration import Homography, load_calibrations, project_observation
from .detector import Detection, RFDETRDetector, SyntheticDetector
from .export import StageTimer, render_annotated_video, write_calibration_json, write_field_view, write_identities_json, write_manifest_json, write_metrics_json, write_observations_csv, write_observations_parquet, write_review_json, write_trajectories_csv
from .identity import TrackletSummary, resolve_teams, stable_anonymous_ids, team_feature_from_crop
from .metrics import config_hash, package_version, sha256_file, summarize_tracks, system_info
from .schema import Observation, RunManifest
from .tracking import IoUTracker, RoboflowTracker, TrackObservation, TrackerAdapter
from .video import ShotBoundary, VideoInfo, detect_shots, iter_video_frames, shot_ranges


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="football-tracking", description="Offline American football player tracking")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run detection, tracking, identity and exports")
    _add_run_arguments(run)
    benchmark = subparsers.add_parser("benchmark", help="run local decode, detector, tracker and export benchmarks")
    _add_run_arguments(benchmark)
    benchmark.add_argument("--model-frames", type=int, default=16, help="maximum frames for an optional real detector timing probe")
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--detector", choices=("rfdetr", "synthetic"), default="rfdetr")
    parser.add_argument("--detector-checkpoint", type=str, default=None)
    parser.add_argument("--model-size", choices=("small", "medium"), default="small")
    parser.add_argument("--tracker", choices=("botsort", "bytetrack", "iou"), default="botsort")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--manual-cut", type=int, action="append", default=[])
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--team-prototypes", type=Path, default=None, help="JSON object mapping team names to RGB triples")
    parser.add_argument("--calibration", type=Path, default=None, help="JSON with image_points and field_points arrays")


def _boundaries_for_video(info: VideoInfo, manual_cuts: Sequence[int]) -> list[ShotBoundary]:
    cuts = sorted({int(cut) for cut in manual_cuts if 0 < int(cut) < info.frame_count})
    if cuts:
        return [ShotBoundary(0, 0, "start", 1.0)] + [ShotBoundary(cut, int(round(cut / info.source_fps / info.time_base[0] * info.time_base[1])), "manual", 1.0) for cut in cuts]
    thumbnails: list[np.ndarray] = []
    source_indices: list[int] = []
    stride = max(1, int(round(info.source_fps / 12.0)))
    for frame_index, _, frame in iter_video_frames(info.path):
        if frame_index % stride == 0:
            thumbnails.append(frame[:: max(1, frame.shape[0] // 72), :: max(1, frame.shape[1] // 128)])
            source_indices.append(frame_index)
    if not thumbnails:
        return []
    candidates = detect_shots(thumbnails, fps=max(1.0, info.source_fps / stride), scene_threshold=0.28)
    boundaries = []
    for candidate in candidates:
        source_index = source_indices[min(candidate.frame_index, len(source_indices) - 1)]
        boundaries.append(ShotBoundary(source_index, int(round(source_index / info.source_fps / info.time_base[0] * info.time_base[1])), candidate.reason, candidate.confidence))
    return boundaries


def _shot_index(frame_index: int, boundaries: Sequence[ShotBoundary]) -> int:
    current = 0
    for index, boundary in enumerate(boundaries):
        if boundary.frame_index <= frame_index:
            current = index
        else:
            break
    return current


def _build_detector(args: argparse.Namespace):
    if args.detector == "synthetic":
        return SyntheticDetector()
    return RFDETRDetector(model_size=args.model_size, checkpoint=args.detector_checkpoint, device=args.device, threshold=args.threshold, cache_dir=args.output / ".roboflow")


def _build_tracker(kind: str, fps: float, shot_id: str, enable_cmc: bool = True) -> TrackerAdapter:
    if kind == "iou":
        tracker = IoUTracker(fps=fps, lost_track_buffer=max(1, int(round(fps))))
        tracker.reset(shot_id)
        return tracker
    return RoboflowTracker(kind=kind, fps=fps, lost_track_buffer=30, enable_cmc=enable_cmc, shot_id=shot_id)


def _load_prototypes(path: Path | None) -> dict[str, Sequence[float]] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("team prototypes must be a JSON object")
    return {str(name): tuple(float(channel) for channel in channels) for name, channels in value.items()}


def _load_calibrations(path: Path | None) -> dict[str, Homography]:
    if path is None:
        return {}
    return load_calibrations(path)


def _is_player(detection: Detection) -> bool:
    name = detection.class_name.lower()
    return detection.class_id == 0 or name in {"player", "person", "football_player", "athlete"}


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    source = args.input
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    info = VideoInfo.from_path(source)
    boundaries = _boundaries_for_video(info, args.manual_cut)
    ranges = shot_ranges(info.frame_count, boundaries)
    run_id = f"run-{sha256_file(source)[:12]}-{args.tracker}"
    detector_config = {"detector": args.detector, "model_size": args.model_size, "checkpoint": args.detector_checkpoint, "threshold": args.threshold}
    detector_hash = config_hash(detector_config)
    cache_path = destination / "detections.jsonl"
    detector = None
    cached: dict[int, list[Detection]] | None = None
    if cache_path.is_file():
        try:
            cached = DetectionCache.load(cache_path, source_hash=sha256_file(source), detector_config_hash=detector_hash)
        except CacheMismatch:
            cached = None
    if cached is None:
        detector = _build_detector(args)
    timer = StageTimer()
    raw_tracks: list[TrackObservation] = []
    tracklet_features: dict[str, list[tuple[float, float, float]]] = {}
    detections_for_cache: dict[int, list[Detection]] = {}
    current_shot = -1
    tracker: TrackerAdapter | None = None
    with timer.stage("decode_and_track"):
        for frame_index, pts, frame in iter_video_frames(source):
            shot_index = _shot_index(frame_index, boundaries)
            shot_id = f"shot-{shot_index}"
            if shot_index != current_shot:
                tracker = _build_tracker(args.tracker, info.source_fps, shot_id)
                current_shot = shot_index
            assert tracker is not None
            if cached is not None:
                detections = cached.get(frame_index, [])
            else:
                with timer.stage("detection"):
                    assert detector is not None
                    detections = detector.predict(frame)
            detections_for_cache[frame_index] = list(detections)
            players = [detection for detection in detections if _is_player(detection)]
            timestamp_s = pts * info.time_base[0] / info.time_base[1]
            with timer.stage("tracking"):
                rows = tracker.update(players, frame, timestamp_s, frame_index=frame_index, pts=pts)
            raw_tracks.extend(rows)
            for row in rows:
                x1, y1, x2, y2 = (int(max(0, round(value))) for value in row.bbox_xyxy_px)
                x1, x2 = min(x1, frame.shape[1] - 1), min(x2, frame.shape[1])
                y1, y2 = min(y1, frame.shape[0] - 1), min(y2, frame.shape[0])
                if x2 > x1 and y2 > y1:
                    feature = team_feature_from_crop(frame[y1:y2, x1:x2])
                    if feature is not None:
                        tracklet_features.setdefault(row.tracklet_id, []).append(feature)
    if cached is None:
        DetectionCache.save(cache_path, sha256_file(source), detector_hash, detections_for_cache)
    timer.timings.setdefault("detection_s", 0.0)
    tracklet_ids = sorted({row.tracklet_id for row in raw_tracks})
    summaries = [TrackletSummary(tracklet_id, tuple(row for row in raw_tracks if row.tracklet_id == tracklet_id), tuple(tracklet_features.get(tracklet_id, ()))) for tracklet_id in tracklet_ids]
    team_prototypes = _load_prototypes(args.team_prototypes)
    teams = resolve_teams(summaries, prototypes=team_prototypes)
    identity_map = stable_anonymous_ids(tracklet_ids, [])
    calibrations = _load_calibrations(args.calibration)
    observations: list[Observation] = []
    for row in raw_tracks:
        evidence = teams.get(row.tracklet_id)
        observation = Observation(
            run_id=run_id,
            shot_id=row.tracklet_id.split(":", 1)[0],
            frame_index=row.frame_index,
            pts=row.pts,
            time_base=info.time_base,
            tracklet_id=row.tracklet_id,
            player_id=identity_map.get(row.tracklet_id),
            bbox_xyxy_px=row.bbox_xyxy_px,
            detection_score=row.score,
            team=evidence.team if evidence else "unknown",
            team_score=evidence.score if evidence else 0.0,
            jersey_number=None,
            field_xy_yards=None,
            position_source=None,
            calibration_id=None,
            identity_version=1,
        )
        homography = calibrations.get(observation.shot_id) or calibrations.get("*")
        if homography is not None:
            observation = project_observation(observation, homography)
        observations.append(observation)
    with timer.stage("export"):
        write_observations_csv(destination / "observations.csv", observations)
        write_observations_parquet(destination / "observations.parquet", observations)
        write_trajectories_csv(destination / "trajectories.csv", observations)
        write_identities_json(destination / "identities.json", identity_map)
        write_metrics_json(destination / "shots.json", {"boundaries": [{"frame_index": boundary.frame_index, "pts": boundary.pts, "reason": boundary.reason, "confidence": boundary.confidence} for boundary in boundaries], "ranges": ranges})
        trajectories: dict[str, list[tuple[float, float]]] = {}
        for observation in observations:
            if observation.player_id and observation.field_xy_yards is not None:
                trajectories.setdefault(observation.player_id, []).append(observation.field_xy_yards)
        write_field_view(destination / "field-view.png", trajectories)
        write_calibration_json(destination / "calibration.json", calibrations)
        write_review_json(destination / "review.json", {"status": "unresolved_cross_view", "shot_count": len(boundaries), "cross_view_identity_resolved": False, "notes": ["Replay relationship remains unresolved unless manually aligned and linked.", "Generic or proxy detector identities are not roster identities."]})
        render_annotated_video(source, destination / "annotated.mp4", observations, boundaries)
    status = "complete_with_unresolved"
    manifest = RunManifest(
        run_id=run_id,
        input_path=str(source),
        input_sha256=sha256_file(source),
        input_duration_s=info.duration_s,
        source_fps=info.source_fps,
        frame_count=info.frame_count,
        detector=args.detector if not getattr(detector, "is_proxy", False) else "synthetic_proxy",
        detector_version="proxy" if getattr(detector, "is_proxy", False) else package_version("rfdetr", "unavailable"),
        tracker=args.tracker,
        tracker_version=package_version("trackers", "fallback" if args.tracker == "iou" else "unavailable"),
        config_hash=detector_hash,
        device=args.device or "auto",
        status=status,
        timings=timer.timings,
        api_usage={"requests": 0, "mode": "disabled"},
    )
    write_manifest_json(destination / "run-manifest.json", manifest)
    write_metrics_json(destination / "metrics.json", {"status": status, "system": system_info(), "track_summary": summarize_tracks(raw_tracks), "observation_count": len(observations), "proxy_detector": bool(getattr(detector, "is_proxy", False)), "detection_cache_hit": cached is not None, "timings": timer.timings})
    return manifest.to_dict()


def _tracker_benchmark(source: Path, info: VideoInfo, boundaries: Sequence[ShotBoundary], kind: str) -> dict[str, Any]:
    timer = StageTimer()
    tracker: TrackerAdapter | None = None
    current_shot = -1
    rows: list[TrackObservation] = []
    proxy = SyntheticDetector()
    try:
        with timer.stage("tracking"):
            for frame_index, pts, frame in iter_video_frames(source):
                shot_index = _shot_index(frame_index, boundaries)
                shot_id = f"shot-{shot_index}"
                if shot_index != current_shot:
                    tracker = _build_tracker(kind, info.source_fps, shot_id)
                    current_shot = shot_index
                assert tracker is not None
                detections = proxy.predict(frame)
                rows.extend(tracker.update(detections, frame, pts * info.time_base[0] / info.time_base[1], frame_index=frame_index, pts=pts))
        return {"status": "complete", "kind": kind, "proxy_detections": True, "tracks": summarize_tracks(rows), "timings": timer.timings, "frames": info.frame_count}
    except Exception as error:
        return {"status": "unavailable", "kind": kind, "error": f"{type(error).__name__}: {error}"}


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    info = VideoInfo.from_path(args.input)
    boundaries = _boundaries_for_video(info, args.manual_cut)
    decode_timer = StageTimer()
    decoded = 0
    first_pts = None
    last_pts = None
    with decode_timer.stage("decode"):
        for frame_index, pts, _ in iter_video_frames(args.input):
            decoded += 1
            first_pts = pts if first_pts is None else first_pts
            last_pts = pts
    detector_report: dict[str, Any]
    detector_timer = StageTimer()
    detector = None
    detector_errors: list[str] = []
    sampled = 0
    with detector_timer.stage("detector"):
        try:
            detector = _build_detector(args)
            for frame_index, _, frame in iter_video_frames(args.input):
                if sampled >= max(0, int(args.model_frames)):
                    break
                detector.predict(frame)
                sampled += 1
        except Exception as error:
            detector_errors.append(f"{type(error).__name__}: {error}")
    if detector_errors:
        detector_report = {"status": "unavailable", "requested": args.detector, "sampled_frames": sampled, "error": detector_errors[0], "timings": detector_timer.timings}
    else:
        detector_report = {"status": "complete", "requested": args.detector, "sampled_frames": sampled, "proxy": bool(getattr(detector, "is_proxy", False)), "timings": detector_timer.timings}
    kinds = [args.tracker] if args.tracker == "iou" else ["bytetrack", "botsort", "iou"]
    tracker_reports = {kind: _tracker_benchmark(args.input, info, boundaries, kind) for kind in kinds}
    identity_fixture = stable_anonymous_ids(["shot-0:t1", "shot-1:t1"], [])
    pipeline_args = argparse.Namespace(**vars(args))
    pipeline_args.detector = "synthetic" if args.detector == "synthetic" or detector_errors else args.detector
    pipeline_args.tracker = args.tracker if tracker_reports.get(args.tracker, {}).get("status") == "complete" else "iou"
    pipeline_args.output = args.output / "pipeline"
    try:
        manifest = run_pipeline(pipeline_args)
        export_report = {"status": "complete", "manifest": manifest, "artifact_count": len(list(pipeline_args.output.iterdir()))}
    except Exception as error:
        export_report = {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}
    report = {
        "status": "complete" if export_report["status"] == "complete" else "incomplete",
        "input": {"path": str(args.input), "sha256": sha256_file(args.input), "duration_s": info.duration_s, "fps": info.source_fps, "metadata_frame_count": info.frame_count, "decoded_frame_count": decoded, "first_pts": first_pts, "last_pts": last_pts, "time_base": list(info.time_base)},
        "decode": {"status": "complete", "timings": decode_timer.timings},
        "detector": detector_report,
        "trackers": tracker_reports,
        "identity": {"status": "complete", "proxy": True, "fixture_ids": identity_fixture, "ground_truth": False},
        "export": export_report,
        "system": system_info(),
    }
    write_metrics_json(args.output / "benchmark.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            result = run_pipeline(args)
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            result = benchmark(args)
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        print(f"football-tracking: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
