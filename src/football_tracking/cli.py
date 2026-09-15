"""Command-line entry points for local football tracking and benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .cache import CacheMismatch, DetectionCache
from .calibration import Homography, load_calibrations, project_observation
from .calibration_timeline import CalibrationTimeline, load_calibration_timeline
from .detector import Detection, RFDETRDetector, SyntheticDetector
from .export import StageTimer, render_annotated_video, write_calibration_json, write_field_view, write_identities_json, write_manifest_json, write_metrics_json, write_observations_csv, write_observations_parquet, write_review_json, write_trajectories_csv
from .identity import IdentityLink, TrackletSummary, resolve_teams, stable_anonymous_ids, team_feature_from_crop
from .identity_resolution import ResolutionPolicy, resolve_play_identities
from .tracklet_refinement import refine_tracklets
from .evaluation import EvaluationError, evaluate_cross_shot_identity, evaluate_tracking, load_cross_shot_identity, load_reviewed_mot_reference
from .metrics import config_hash, package_version, sha256_file, summarize_tracks, system_info
from .memory import MemoryBudget
from .replay import FieldTrack, PlayAlignment, ReplayAlignmentError, load_play_alignment
from .schema import Observation, RunManifest
from .tracking import IoUTracker, McByteTracker, RoboflowTracker, TrackObservation, TrackerAdapter
from .video import ShotBoundary, VideoInfo, detect_shots, iter_video_frames, shot_ranges


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="football-tracking", description="Offline American football player tracking")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run detection, tracking, identity and exports")
    _add_run_arguments(run)
    benchmark = subparsers.add_parser("benchmark", help="run local decode, detector, tracker and export benchmarks")
    _add_run_arguments(benchmark)
    benchmark.add_argument("--model-frames", type=int, default=16, help="maximum frames for an optional real detector timing probe")
    cache = subparsers.add_parser("cache", help="build a bounded native-frame detector-cache chunk")
    _add_cache_arguments(cache)
    merge = subparsers.add_parser("merge-cache", help="strictly merge detector-cache chunks into a full replay cache")
    _add_cache_arguments(merge)
    merge.add_argument("--part", type=Path, action="append", required=True, help="cache chunk to merge; repeat in source-frame order")
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--detector", choices=("rfdetr", "synthetic"), default="rfdetr")
    parser.add_argument("--detector-checkpoint", type=str, default=None)
    parser.add_argument("--model-size", choices=("small", "medium"), default="small")
    parser.add_argument("--tracker", choices=("botsort", "bytetrack", "mcbyte", "iou"), default="botsort")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mcbyte-device", choices=("cpu", "cuda", "mps"), default="cpu", help="explicit device for McByte mask models")
    parser.add_argument("--mcbyte-sam-checkpoint", type=Path, default=None, help="existing SAM vit_b checkpoint; no download is attempted")
    parser.add_argument("--mcbyte-cutie-checkpoint", type=Path, default=None, help="existing Cutie base-mega checkpoint; no download is attempted")
    parser.add_argument("--mcbyte-masks", choices=("on", "off"), default="on", help="request full mask-assisted McByte or its supported mask-free ablation")
    parser.add_argument("--botsort-cmc", choices=("on", "off"), default="on", help="enable BoT-SORT camera-motion compensation")
    parser.add_argument("--manual-cut", type=int, action="append", default=[])
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-memory-mb", type=float, default=2048.0, help="host-process peak RSS budget; fail early above this value (default: 2048)")
    parser.add_argument("--allow-unbounded-memory", action="store_true", help="explicitly allow mask-enabled model construction when the host cannot enforce the RSS ceiling")
    parser.add_argument("--detection-cache", type=Path, default=None, help="strict shared detector cache for controlled tracker replay")
    parser.add_argument("--detector-class-mapping", type=Path, default=None, help="JSON class-id mapping required to validate a controlled shared cache")
    parser.add_argument("--reviewed-reference", type=Path, default=None, help="reviewed MOT-style reference JSON; required for quality scoring")
    parser.add_argument("--team-prototypes", type=Path, default=None, help="JSON object mapping team names to RGB triples")
    parser.add_argument("--calibration", type=Path, default=None, help="legacy landmark JSON or schema-v2 PTS-scoped calibration timeline")
    parser.add_argument("--play-alignment", type=Path, default=None, help="reviewed snap-anchor JSON enabling constrained cross-shot identity joins")
    parser.add_argument("--reviewed-splits", type=Path, default=None, help="reviewed split overlay JSON for within-shot tracklet purity")
    parser.add_argument("--identity-threshold", type=float, default=0.65, help="minimum cross-shot identity score")
    parser.add_argument("--identity-margin", type=float, default=0.1, help="minimum global assignment ambiguity margin")
    parser.add_argument("--identity-max-distance-yards", type=float, default=6.0, help="maximum robust field distance for a candidate")
    parser.add_argument("--identity-min-overlap-duration-s", type=float, default=0.4, help="minimum aligned temporal support for a candidate")
    parser.add_argument("--start-frame", type=int, default=0, help="inclusive source-frame start for a bounded analysis window")
    parser.add_argument("--end-frame", type=int, default=None, help="exclusive source-frame end for a bounded analysis window")


def _add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--detector", choices=("rfdetr", "synthetic"), default="rfdetr")
    parser.add_argument("--detector-checkpoint", type=str, default=None)
    parser.add_argument("--model-size", choices=("small", "medium"), default="small")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-memory-mb", type=float, default=2048.0)
    parser.add_argument("--detector-class-mapping", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="exclusive source-frame end for a cache chunk")


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
    output = Path(args.output)
    cache_dir = output.parent / ".roboflow" if getattr(args, "command", None) == "cache" else output / ".roboflow"
    return RFDETRDetector(model_size=args.model_size, checkpoint=args.detector_checkpoint, device=args.device, threshold=args.threshold, cache_dir=cache_dir)


def _build_tracker(
    kind: str,
    fps: float,
    shot_id: str,
    enable_cmc: bool = True,
    mcbyte_device: str = "cpu",
    mcbyte_sam_checkpoint: Path | None = None,
    mcbyte_cutie_checkpoint: Path | None = None,
    mcbyte_masks: str = "on",
) -> TrackerAdapter:
    if kind == "iou":
        tracker = IoUTracker(fps=fps, lost_track_buffer=max(1, int(round(fps))))
        tracker.reset(shot_id)
        return tracker
    if kind == "mcbyte":
        # Cutie loads a ResNet backbone through torch.hub. Keep that auxiliary
        # asset inside ignored evaluation artifacts instead of touching a
        # user-home cache or downloading during an otherwise reproducible run.
        os.environ.setdefault("TORCH_HOME", str(Path("artifacts/mcbyte-evaluation/torch-cache").resolve()))
        return McByteTracker(
            fps=fps,
            device=mcbyte_device,
            enable_masks=mcbyte_masks == "on",
            sam_checkpoint=mcbyte_sam_checkpoint,
            cutie_checkpoint=mcbyte_cutie_checkpoint,
            shot_id=shot_id,
        )
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
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and int(value.get("schema_version", 1)) >= 2:
        return {}
    return load_calibrations(path)


def _load_calibration_timeline(path: Path | None) -> CalibrationTimeline | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and int(value.get("schema_version", 1)) >= 2:
        return load_calibration_timeline(path)
    return None


def _load_class_mapping(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("detector class mapping must be a non-empty JSON object")
    return {str(key): str(name) for key, name in value.items()}


def _load_reviewed_splits(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read reviewed splits: {error}") from error
    if not isinstance(value, dict) or value.get("reviewed") is not True or not isinstance(value.get("splits"), dict):
        raise ValueError("reviewed splits must be marked reviewed: true and contain splits")
    result: dict[str, list[dict[str, Any]]] = {}
    for tracklet_id, specs in value["splits"].items():
        if not isinstance(specs, list):
            raise ValueError(f"reviewed split for {tracklet_id} must be a list")
        result[str(tracklet_id)] = [dict(spec) for spec in specs if isinstance(spec, dict)]
        if len(result[str(tracklet_id)]) != len(specs):
            raise ValueError(f"reviewed split for {tracklet_id} contains an invalid record")
    return result


def _is_player(detection: Detection) -> bool:
    name = detection.class_name.lower()
    return detection.class_id == 0 or name in {"player", "person", "football_player", "athlete"}


def _cache_identity(args: argparse.Namespace, source: Path) -> tuple[str, str, str | None, dict[str, str], dict[str, Any]]:
    source_hash = sha256_file(source)
    checkpoint_hash = sha256_file(args.detector_checkpoint) if args.detector_checkpoint and Path(args.detector_checkpoint).is_file() else None
    detector_config = {"detector": args.detector, "model_size": args.model_size, "checkpoint_sha256": checkpoint_hash, "threshold": args.threshold, "player_filter": "project_player_v1"}
    class_mapping = _load_class_mapping(args.detector_class_mapping)
    assert class_mapping is not None
    provenance = {"checkpoint_sha256": checkpoint_hash, "class_mapping": class_mapping, "proxy": args.detector == "synthetic"}
    return source_hash, config_hash(detector_config), checkpoint_hash, class_mapping, provenance


def build_cache_chunk(args: argparse.Namespace) -> dict[str, Any]:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    info = VideoInfo.from_path(args.input)
    start = int(args.start_frame)
    end = info.frame_count if args.end_frame is None else int(args.end_frame)
    if not 0 <= start < end <= info.frame_count:
        raise ValueError(f"cache range must be within [0, {info.frame_count}] and non-empty")
    source_hash, detector_hash, _, class_mapping, provenance = _cache_identity(args, args.input)
    budget = MemoryBudget(args.max_memory_mb)
    budget.enforce_process_limit()
    detector = _build_detector(args)
    budget.sample("detector_initialization")
    frames: dict[int, list[Detection]] = {}
    for frame_index, _, frame in iter_video_frames(args.input, start_frame=start, end_frame=end):
        frames[frame_index] = [detection for detection in detector.predict(frame) if _is_player(detection)]
        if frame_index % 30 == 0:
            budget.sample("detection")
    observed_mapping = {str(detection.class_id): detection.class_name for rows in frames.values() for detection in rows}
    if any(class_mapping.get(key) != value for key, value in observed_mapping.items()):
        raise ValueError("observed detector class mapping does not match --detector-class-mapping")
    budget.sample("detection_complete")
    DetectionCache.save(args.output, source_hash, detector_hash, frames, provenance=provenance)
    return {"status": "complete", "output": str(args.output), "frame_start": start, "frame_end": end, "frame_count": len(frames), "source_hash": source_hash, "detector_config_hash": detector_hash, "provenance": provenance, "peak_rss_mb": budget.peak_mb}


def merge_cache_chunks(args: argparse.Namespace) -> dict[str, Any]:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    info = VideoInfo.from_path(args.input)
    source_hash, detector_hash, _, _, provenance = _cache_identity(args, args.input)
    merged = DetectionCache.merge_strict_parts(args.output, args.part, source_hash, detector_hash, frame_count=info.frame_count, provenance=provenance)
    return {"status": "complete", "output": str(args.output), "frame_count": len(merged), "source_hash": source_hash, "detector_config_hash": detector_hash, "provenance": provenance}


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    source = args.input
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    info = VideoInfo.from_path(source)
    process_start = int(args.start_frame)
    process_end = info.frame_count if args.end_frame is None else int(args.end_frame)
    if not 0 <= process_start < process_end <= info.frame_count:
        raise ValueError(f"analysis window must be within [0, {info.frame_count}] and non-empty")
    boundaries = _boundaries_for_video(info, args.manual_cut)
    ranges = shot_ranges(info.frame_count, boundaries)
    alignment: PlayAlignment | None = load_play_alignment(args.play_alignment) if args.play_alignment else None
    if alignment is not None:
        for anchor in alignment.anchors:
            expected_shot = f"shot-{_shot_index(anchor.source_frame, boundaries)}"
            if expected_shot != anchor.shot_id:
                raise ReplayAlignmentError(f"alignment anchor {anchor.shot_id}:{anchor.source_frame} falls in {expected_shot}")
    source_hash = sha256_file(source)
    checkpoint_hash = sha256_file(args.detector_checkpoint) if args.detector_checkpoint and Path(args.detector_checkpoint).is_file() else None
    calibration_hash = sha256_file(args.calibration) if args.calibration and args.calibration.is_file() else None
    alignment_hash = sha256_file(args.play_alignment) if args.play_alignment and args.play_alignment.is_file() else None
    prototypes_hash = sha256_file(args.team_prototypes) if args.team_prototypes and args.team_prototypes.is_file() else None
    reference_hash = sha256_file(args.reviewed_reference) if args.reviewed_reference and args.reviewed_reference.is_file() else None
    split_hash = sha256_file(args.reviewed_splits) if args.reviewed_splits and args.reviewed_splits.is_file() else None
    tracker_config = {
        "tracker": args.tracker,
        "mcbyte_device": args.mcbyte_device if args.tracker == "mcbyte" else None,
        "mcbyte_masks": args.mcbyte_masks if args.tracker == "mcbyte" else None,
        "sam_checkpoint_sha256": sha256_file(args.mcbyte_sam_checkpoint) if args.mcbyte_sam_checkpoint and args.mcbyte_sam_checkpoint.is_file() else None,
        "cutie_checkpoint_sha256": sha256_file(args.mcbyte_cutie_checkpoint) if args.mcbyte_cutie_checkpoint and args.mcbyte_cutie_checkpoint.is_file() else None,
    }
    tracker_hash = config_hash(tracker_config)
    detector_config = {"detector": args.detector, "model_size": args.model_size, "checkpoint_sha256": checkpoint_hash, "threshold": args.threshold, "player_filter": "project_player_v1"}
    detector_hash = config_hash(detector_config)
    analysis_config = {
        "schema_version": 2,
        "detector_hash": detector_hash,
        "tracker_hash": tracker_hash,
        "calibration_sha256": calibration_hash,
        "alignment_sha256": alignment_hash,
        "team_prototypes_sha256": prototypes_hash,
        "reviewed_splits_sha256": split_hash,
        "manual_cuts": sorted(int(cut) for cut in args.manual_cut),
        "boundaries": [{"frame_index": boundary.frame_index, "pts": boundary.pts, "reason": boundary.reason, "confidence": boundary.confidence} for boundary in boundaries],
        "window": {"start_frame": process_start, "end_frame": process_end},
        "resolver_policy": {"threshold": args.identity_threshold, "margin": args.identity_margin, "max_field_distance_yards": args.identity_max_distance_yards, "min_overlap_samples": 5, "min_overlap_duration_s": args.identity_min_overlap_duration_s, "sample_tolerance_s": 0.05},
        "contact_policy": {"source": "bottom_center", "version": 1},
    }
    analysis_hash = config_hash(analysis_config)
    run_id = f"run-{source_hash[:12]}-{args.tracker}-{analysis_hash[:8]}"
    cache_path = args.detection_cache or destination / "detections.jsonl"
    strict_shared_cache = args.detection_cache is not None
    memory_budget = MemoryBudget(None if args.allow_unbounded_memory else args.max_memory_mb)
    memory_budget.enforce_process_limit()
    if args.tracker == "mcbyte" and args.mcbyte_masks == "on" and not memory_budget.hard_limit_applied and not args.allow_unbounded_memory:
        raise RuntimeError("macOS cannot enforce the requested host memory ceiling before McByte mask construction; rerun with --allow-unbounded-memory only on a separately monitored host")
    configured_class_mapping = _load_class_mapping(args.detector_class_mapping)
    if strict_shared_cache and configured_class_mapping is None:
        raise ValueError("--detection-cache requires --detector-class-mapping to validate the detector class mapping")
    expected_provenance = {"checkpoint_sha256": checkpoint_hash, "class_mapping": configured_class_mapping, "proxy": args.detector == "synthetic"}
    detector = None
    detector_is_proxy = args.detector == "synthetic"
    cached: dict[int, list[Detection]] | None = None
    if cache_path.is_file():
        try:
            cached = (
                DetectionCache.load_strict(cache_path, source_hash, detector_hash, frame_count=info.frame_count, provenance=expected_provenance)
                if strict_shared_cache else DetectionCache.load(cache_path, source_hash=source_hash, detector_config_hash=detector_hash)
            )
            if cached is not None and not set(range(process_start, process_end)).issubset(cached):
                cached = None
        except CacheMismatch:
            if strict_shared_cache:
                raise
            cached = None
    elif strict_shared_cache:
        raise FileNotFoundError(f"shared detection cache does not exist: {cache_path}")
    if cached is None:
        detector = _build_detector(args)
        detector_is_proxy = bool(getattr(detector, "is_proxy", False))
        memory_budget.sample("detector_initialization")
    timer = StageTimer()
    raw_tracks: list[TrackObservation] = []
    tracklet_features: dict[str, list[tuple[float, float, float]]] = {}
    detections_for_cache: dict[int, list[Detection]] = {}
    current_shot = -1
    tracker: TrackerAdapter | None = None
    tracker_modes: list[dict[str, Any]] = []
    with timer.stage("decode_and_track"):
        for frame_index, pts, frame in iter_video_frames(source, start_frame=process_start, end_frame=process_end):
            shot_index = _shot_index(frame_index, boundaries)
            shot_id = f"shot-{shot_index}"
            if shot_index != current_shot:
                if tracker is not None and isinstance(tracker, McByteTracker):
                    tracker_modes.append(tracker.effective_mode())
                tracker = _build_tracker(
                    args.tracker, info.source_fps, shot_id, enable_cmc=args.botsort_cmc == "on",
                    mcbyte_device=args.mcbyte_device,
                    mcbyte_sam_checkpoint=args.mcbyte_sam_checkpoint,
                    mcbyte_cutie_checkpoint=args.mcbyte_cutie_checkpoint,
                    mcbyte_masks=args.mcbyte_masks,
                )
                memory_budget.sample("tracker_initialization")
                current_shot = shot_index
            assert tracker is not None
            if cached is not None:
                detections = cached.get(frame_index, [])
            else:
                with timer.stage("detection"):
                    assert detector is not None
                    detections = detector.predict(frame)
            players = [detection for detection in detections if _is_player(detection)]
            detections_for_cache[frame_index] = list(players)
            timestamp_s = pts * info.time_base[0] / info.time_base[1]
            with timer.stage("tracking"):
                rows = tracker.update(players, frame, timestamp_s, frame_index=frame_index, pts=pts)
            raw_tracks.extend(rows)
            if frame_index % 30 == 0:
                memory_budget.sample("decode_and_track")
            for row in rows:
                x1, y1, x2, y2 = (int(max(0, round(value))) for value in row.bbox_xyxy_px)
                x1, x2 = min(x1, frame.shape[1] - 1), min(x2, frame.shape[1])
                y1, y2 = min(y1, frame.shape[0] - 1), min(y2, frame.shape[0])
                if x2 > x1 and y2 > y1:
                    feature = team_feature_from_crop(frame[y1:y2, x1:x2])
                    if feature is not None:
                        tracklet_features.setdefault(row.tracklet_id, []).append(feature)
    if isinstance(tracker, McByteTracker):
        tracker_modes.append(tracker.effective_mode())
    memory_budget.sample("tracking_complete")
    if cached is None:
        class_mapping = {str(detection.class_id): detection.class_name for rows in detections_for_cache.values() for detection in rows}
        if configured_class_mapping is not None and any(configured_class_mapping.get(key) != value for key, value in class_mapping.items()):
            raise ValueError("observed detector class mapping does not match --detector-class-mapping")
        class_mapping = configured_class_mapping or class_mapping or {"0": "player"}
        DetectionCache.save(
            cache_path,
            source_hash,
            detector_hash,
            detections_for_cache,
            provenance={"checkpoint_sha256": checkpoint_hash, "class_mapping": class_mapping or {"0": "player"}, "proxy": detector_is_proxy, "frame_start": process_start, "frame_end": process_end},
        )
        (destination / "detector-class-mapping.json").write_text(json.dumps(class_mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    timer.timings.setdefault("detection_s", 0.0)
    reviewed_splits = _load_reviewed_splits(args.reviewed_splits)
    refined_segments = refine_tracklets(raw_tracks, reviewed_splits or None)
    segment_for_observation: dict[tuple[str, int], str] = {}
    for segment in refined_segments:
        for row in segment.observations:
            key = (row.tracklet_id, row.frame_index)
            if key in segment_for_observation:
                raise ValueError(f"reviewed splits assign one observation to multiple segments: {key}")
            segment_for_observation[key] = segment.segment_id
    derived_tracklet_ids = sorted(set(segment_for_observation.values()))
    source_for_segment = {segment.segment_id: segment.source_tracklet_id for segment in refined_segments}
    summaries = [
        TrackletSummary(
            tracklet_id,
            tuple(row for row in raw_tracks if segment_for_observation.get((row.tracklet_id, row.frame_index), row.tracklet_id) == tracklet_id),
            tuple(tracklet_features.get(source_for_segment.get(tracklet_id, tracklet_id), ())),
        )
        for tracklet_id in derived_tracklet_ids
    ]
    tracklet_ids = derived_tracklet_ids
    identity_tracks = [
        TrackObservation(segment_for_observation.get((row.tracklet_id, row.frame_index), row.tracklet_id), row.frame_index, row.pts, row.bbox_xyxy_px, row.score, row.state)
        for row in raw_tracks
    ]
    team_prototypes = _load_prototypes(args.team_prototypes)
    teams = resolve_teams(summaries, prototypes=team_prototypes)
    identity_links: list[IdentityLink] = []
    link_report: dict[str, Any] = {"status": "not_attempted", "reason": "--play-alignment was not provided"}
    calibrations = _load_calibrations(args.calibration)
    calibration_timeline = _load_calibration_timeline(args.calibration)
    observations: list[Observation] = []
    for row in raw_tracks:
        identity_tracklet_id = segment_for_observation.get((row.tracklet_id, row.frame_index), row.tracklet_id)
        source_tracklet_id = row.tracklet_id if identity_tracklet_id != row.tracklet_id else None
        evidence = teams.get(identity_tracklet_id) or teams.get(row.tracklet_id)
        shot_id = row.tracklet_id.split(":", 1)[0]
        observation = Observation(
            run_id=run_id,
            shot_id=shot_id,
            frame_index=row.frame_index,
            pts=row.pts,
            time_base=info.time_base,
            tracklet_id=identity_tracklet_id,
            player_id=None,
            bbox_xyxy_px=row.bbox_xyxy_px,
            detection_score=row.score,
            team=evidence.team if evidence else "unknown",
            team_score=evidence.score if evidence else 0.0,
            jersey_number=None,
            field_xy_yards=None,
            position_source=None,
            calibration_id=None,
            identity_version=2,
            play_id=alignment.play_id if alignment and shot_id in alignment.shots() else None,
            source_tracklet_id=source_tracklet_id,
        )
        estimate = calibration_timeline.at(observation.shot_id, observation.pts) if calibration_timeline is not None else None
        homography = estimate.homography if estimate is not None else (calibrations.get(observation.shot_id) or calibrations.get("*"))
        if homography is not None:
            observation = project_observation(
                observation,
                homography,
                calibration_status=estimate.status if estimate is not None else homography.status,
                calibration_reason=estimate.reason if estimate is not None else homography.reason,
            )
        observations.append(observation)
    if alignment is not None:
        observations = [
            replace(
                observation,
                play_time_s=alignment.play_time_at_pts(observation.shot_id, observation.pts, info.time_base, info.source_fps, observation.frame_index),
                time_map_id="play-time-map" if alignment.time_map is not None else "legacy-anchor-offset",
            )
            for observation in observations
        ]
        field_tracks: dict[str, list[FieldTrack]] = {}
        samples: dict[str, list[tuple[float, float, float]]] = {}
        for observation in observations:
            if observation.field_xy_yards is None or observation.calibration_status == "invalid":
                continue
            play_time = observation.play_time_s
            if play_time is None:
                continue
            samples.setdefault(observation.tracklet_id, []).append((play_time, *observation.field_xy_yards))
        for tracklet_id, rows in samples.items():
            shot_id = tracklet_id.split(":", 1)[0]
            field_tracks.setdefault(shot_id, []).append(FieldTrack(tracklet_id, shot_id, tuple(sorted(rows))))
        aligned_shots = list(alignment.shots())
        calibrated_shots = set(calibrations) | (set(calibration_timeline.identity_eligible_shots()) if calibration_timeline is not None else set())
        present_shots = [shot for shot in aligned_shots if field_tracks.get(shot)]
        if len(present_shots) < 2:
            link_report = {"status": "abstained", "reason": "no calibrated field positions for the aligned shots", "unresolved_shots": [shot for shot in aligned_shots if shot not in present_shots]}
        elif any(shot not in calibrated_shots for shot in present_shots):
            # A "*" shared-homography fallback is not shot-specific: applying one
            # camera pose's transform to another shot would make "field position"
            # a restatement of image coordinates, which is exactly the
            # image-coordinate merging this design forbids across a cut.
            link_report = {"status": "abstained", "reason": "cross-shot resolution requires shot-specific calibration"}
        else:
            resolution = resolve_play_identities(
                {"play_id": alignment.play_id, "shot_ids": tuple(aligned_shots)},
                field_tracks,
                calibration=calibration_timeline or calibrations,
                time_map=alignment.time_map,
                teams=teams,
                policy=ResolutionPolicy(threshold=args.identity_threshold, margin=args.identity_margin, max_field_distance_yards=args.identity_max_distance_yards, min_overlap_duration_s=args.identity_min_overlap_duration_s),
            )
            identity_links = [link for link in resolution.links if link.decision == "same"]
            link_report = resolution.to_dict()
            link_report["status"] = resolution.status
            link_report["reason"] = resolution.reason
            link_report["accepted_links"] = len(identity_links)
            if len(aligned_shots) <= 2:
                link_report["note"] = "only the first two aligned shots are resolved in this milestone"
    identity_map = stable_anonymous_ids(tracklet_ids, identity_links)
    observations = [
        replace(observation, player_id=identity_map.get(observation.tracklet_id))
        for observation in observations
    ]
    by_player: dict[str, set[str]] = {}
    for tracklet_id, player_id in identity_map.items():
        by_player.setdefault(player_id, set()).add(tracklet_id.split(":", 1)[0])
    link_report.update({
        "shot_count": len(boundaries),
        "tracklet_count": len(tracklet_ids),
        "player_id_count": len(set(identity_map.values())),
        "cross_shot_player_ids": sum(1 for shots in by_player.values() if len(shots) > 1),
        "links": link_report.get("links", [
            {"left_key": link.left_key, "right_key": link.right_key, "decision": link.decision, "score": link.score, "alternative_score": link.alternative_score, "ambiguity_margin": link.ambiguity_margin, "rejection_reason": link.rejection_reason, "evidence_keys": list(link.evidence_keys)}
            for link in identity_links
        ]),
    })
    with timer.stage("export"):
        write_observations_csv(destination / "observations.csv", observations)
        write_observations_parquet(destination / "observations.parquet", observations)
        write_trajectories_csv(destination / "trajectories.csv", observations)
        write_identities_json(destination / "identities.json", identity_map)
        write_metrics_json(destination / "identity-links.json", link_report)
        write_metrics_json(destination / "shots.json", {"boundaries": [{"frame_index": boundary.frame_index, "pts": boundary.pts, "reason": boundary.reason, "confidence": boundary.confidence} for boundary in boundaries], "ranges": ranges})
        trajectories: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for observation in observations:
            if observation.player_id and observation.field_xy_yards is not None:
                trajectories.setdefault((observation.player_id, observation.shot_id), []).append(observation.field_xy_yards)
        write_field_view(destination / "field-view.png", trajectories)
        if calibration_timeline is not None:
            write_metrics_json(destination / "calibration.json", calibration_timeline.to_dict())
        else:
            write_calibration_json(destination / "calibration.json", calibrations)
        resolved = link_report.get("status") == "resolved"
        write_review_json(destination / "review.json", {
            "status": "resolved_cross_view" if resolved else "unresolved_cross_view",
            "shot_count": len(boundaries),
            "cross_view_identity_resolved": resolved,
            "cross_shot_links": link_report.get("accepted_links", 0),
            "notes": [
                "Replay relationship resolved: identity-links.json records the accepted cross-shot"
                " tracklet pairs from the reviewed play alignment."
                if resolved
                else "Replay relationship remains unresolved unless manually aligned and linked.",
                "Generic or proxy detector identities are not roster identities.",
            ],
        })
        render_annotated_video(source, destination / "annotated.mp4", observations, boundaries, start_frame=process_start, end_frame=process_end)
    memory_budget.sample("export_complete")
    quality_report: dict[str, Any]
    if args.reviewed_reference is None:
        quality_report = {"status": "not_evaluated", "reason": "--reviewed-reference was not provided"}
    else:
        try:
            reviewed_reference = load_reviewed_mot_reference(args.reviewed_reference)
            quality_report = evaluate_tracking(raw_tracks, reviewed_reference)
            quality_report["reference_path"] = str(args.reviewed_reference)
            quality_report["reference_sha256"] = sha256_file(args.reviewed_reference)
            quality_report["cross_shot"] = evaluate_cross_shot_identity(
                identity_map,
                identity_tracks,
                reviewed_reference,
                load_cross_shot_identity(args.reviewed_reference),
            )
        except EvaluationError as error:
            quality_report = {"status": "not_evaluated", "reason": f"{type(error).__name__}: {error}"}
    write_metrics_json(destination / "tracking-evaluation.json", quality_report)
    status = "complete" if resolved else "complete_with_unresolved"
    timer.timings["peak_rss_mb"] = memory_budget.peak_mb
    manifest = RunManifest(
        run_id=run_id,
        input_path=str(source),
        input_sha256=sha256_file(source),
        input_duration_s=info.duration_s,
        source_fps=info.source_fps,
        frame_count=info.frame_count,
        detector=args.detector if not detector_is_proxy else "synthetic_proxy",
        detector_version="proxy" if detector_is_proxy else package_version("rfdetr", "unavailable"),
        tracker=args.tracker,
        tracker_version=package_version("trackers", "fallback" if args.tracker == "iou" else "unavailable"),
        config_hash=analysis_hash,
        device=args.device or "auto",
        status=status,
        timings=timer.timings,
        api_usage={"requests": 0, "mode": "disabled"},
    )
    write_manifest_json(destination / "run-manifest.json", manifest)
    write_metrics_json(destination / "metrics.json", {
        "status": status,
        "system": system_info(),
        "track_summary": summarize_tracks(raw_tracks),
        "observation_count": len(observations),
        "proxy_detector": detector_is_proxy,
        "detection_cache_hit": cached is not None,
        "detector_cache": {"path": str(cache_path), "strict_shared": strict_shared_cache, "provenance": expected_provenance},
        "analysis": {"schema_version": 2, "analysis_hash": analysis_hash, "calibration_sha256": calibration_hash, "alignment_sha256": alignment_hash, "team_prototypes_sha256": prototypes_hash, "reviewed_splits_sha256": split_hash, "reference_sha256": reference_hash, "boundary_count": len(boundaries)},
        "window": {"start_frame": process_start, "end_frame": process_end, "processed_frame_count": process_end - process_start, "source_frame_count": info.frame_count},
        "tracker_config": tracker_config,
        "mcbyte": tracker_modes or None,
        "memory": {"budget_mb": memory_budget.limit_mb, "peak_rss_mb": memory_budget.peak_mb, "samples_mb": memory_budget.samples},
        "quality": {"status": quality_report["status"]},
        "timings": timer.timings,
    })
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
    if args.detection_cache is None:
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
    else:
        detector_report = {"status": "controlled_cache", "requested": args.detector, "cache": str(args.detection_cache), "sampled_frames": 0}
    if args.detection_cache is not None:
        pass
    elif detector_errors:
        detector_report = {"status": "unavailable", "requested": args.detector, "sampled_frames": sampled, "error": detector_errors[0], "timings": detector_timer.timings}
    else:
        detector_report = {"status": "complete", "requested": args.detector, "sampled_frames": sampled, "proxy": bool(getattr(detector, "is_proxy", False)), "timings": detector_timer.timings}
    kinds = [args.tracker] if args.tracker == "iou" else ["bytetrack", "botsort", "iou"]
    tracker_reports = {kind: _tracker_benchmark(args.input, info, boundaries, kind) for kind in kinds}
    identity_fixture = stable_anonymous_ids(["shot-0:t1", "shot-1:t1"], [])
    pipeline_args = argparse.Namespace(**vars(args))
    pipeline_args.detector = "synthetic" if args.detection_cache is None and (args.detector == "synthetic" or detector_errors) else args.detector
    pipeline_args.tracker = args.tracker if tracker_reports.get(args.tracker, {}).get("status") == "complete" else "iou"
    pipeline_args.output = args.output / "pipeline"
    try:
        manifest = run_pipeline(pipeline_args)
        export_report = {"status": "complete", "manifest": manifest, "artifact_count": len(list(pipeline_args.output.iterdir()))}
    except Exception as error:
        export_report = {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}
    comparison: dict[str, Any] = {"status": "not_requested", "reason": "--detection-cache is required for controlled real-cache comparison"}
    if args.detection_cache is not None:
        comparison = {"status": "incomplete", "shared_cache": str(args.detection_cache), "variants": {}}
        variants = (
            ("bytetrack", {"tracker": "bytetrack"}),
            ("botsort-cmc-on", {"tracker": "botsort", "botsort_cmc": "on"}),
            ("botsort-cmc-off", {"tracker": "botsort", "botsort_cmc": "off"}),
            ("mcbyte-mask-on", {"tracker": "mcbyte", "mcbyte_masks": "on"}),
            ("mcbyte-mask-off", {"tracker": "mcbyte", "mcbyte_masks": "off"}),
        )
        for name, changes in variants:
            variant_args = argparse.Namespace(**vars(args))
            for key, value in changes.items():
                setattr(variant_args, key, value)
            variant_args.output = args.output / "variants" / name
            try:
                variant_manifest = run_pipeline(variant_args)
                metrics = json.loads((variant_args.output / "metrics.json").read_text(encoding="utf-8"))
                evaluation = json.loads((variant_args.output / "tracking-evaluation.json").read_text(encoding="utf-8"))
                comparison["variants"][name] = {"status": "complete", "manifest": variant_manifest, "metrics": metrics, "evaluation": evaluation}
            except Exception as error:
                comparison["variants"][name] = {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}
        completed = comparison["variants"].values()
        if all(item["status"] == "complete" for item in completed):
            comparison["status"] = "complete"
    report = {
        "status": "complete" if export_report["status"] == "complete" else "incomplete",
        "input": {"path": str(args.input), "sha256": sha256_file(args.input), "duration_s": info.duration_s, "fps": info.source_fps, "metadata_frame_count": info.frame_count, "decoded_frame_count": decoded, "first_pts": first_pts, "last_pts": last_pts, "time_base": list(info.time_base)},
        "decode": {"status": "complete", "timings": decode_timer.timings},
        "detector": detector_report,
        "trackers": tracker_reports,
        "identity": {"status": "complete", "proxy": True, "fixture_ids": identity_fixture, "ground_truth": False},
        "export": export_report,
        "comparison": comparison,
        "system": system_info(),
    }
    write_metrics_json(args.output / "benchmark.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            result = run_pipeline(args)
        elif args.command == "benchmark":
            result = benchmark(args)
        elif args.command == "cache":
            result = build_cache_chunk(args)
        else:
            result = merge_cache_chunks(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, MemoryError, RuntimeError, ValueError, KeyError, ReplayAlignmentError) as error:
        print(f"football-tracking: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
