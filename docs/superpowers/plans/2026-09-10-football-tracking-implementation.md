# American Football Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a reproducible local American-football tracking CLI that detects players with RF-DETR, tracks them per shot with Roboflow BoT-SORT and ByteTrack baselines, resolves teams and anonymous identities, exports video/tables/field trajectories, and records local benchmark evidence.

**Architecture:** A dependency-light core owns immutable video observations, shot boundaries, identity constraints, calibration, and exports. Optional Roboflow adapters load RF-DETR and `trackers` only at runtime, so unit tests remain deterministic and do not need model weights. A batch CLI caches detections, resets trackers at cuts, and instruments each pipeline stage. The first real run uses the supplied clip and a declared checkpoint; if model access is unavailable, the benchmark records that as an explicit missing gate rather than fabricating results.

**Tech Stack:** Python 3.11+, NumPy, OpenCV, SciPy, PyAV/FFmpeg-compatible decoding, RF-DETR 1.10.1, Roboflow trackers 2.6.0, supervision 0.30.2, pytest.

**Spec:** `docs/system-design.md` and `docs/evaluation-plan.md`.

## Global Constraints

- Preserve original frame indices, PTS, time base, and source coordinates.
- Process each shot with a fresh tracker; the sample cut is frame 712 at approximately 11.878533 seconds.
- Pass RGB NumPy images to RF-DETR and BGR frames to BoT-SORT camera-motion compensation.
- Use `trackers.ByteTrackTracker` and `trackers.BoTSORTTracker`; do not use deprecated `supervision.ByteTrack`.
- Keep `tracklet_id`, stable anonymous `player_id`, team evidence, and optional jersey observations separate.
- Never merge replay views or fill long occlusions without evidence; unresolved identities remain explicit.
- Field coordinates are valid only with a checked homography and are measured in a fixed yard coordinate frame.
- API use is optional, bounded, cached, and disabled by default. No external upload is required for the local deliverable.
- Run tests before implementation slices and run full verification before claiming completion.

---

### Task 1: Project bootstrap and contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/football_tracking/__init__.py`
- Create: `src/football_tracking/schema.py`
- Create: `tests/test_schema.py`

**Interfaces:**
- `Observation` is a frozen dataclass with `run_id`, `shot_id`, `frame_index`, `pts`, `time_base`, `tracklet_id`, optional `player_id`, `bbox_xyxy_px`, `detection_score`, `team`, `team_score`, optional `jersey_number`, optional `field_xy_yards`, `position_source`, `calibration_id`, and `identity_version`.
- `RunManifest` serializes input hash, versions, configuration, device, and timing.
- `Observation.to_dict()` emits JSON/CSV-safe scalar values and nullable fields.

- [x] Write schema serialization and required-field tests.
- [x] Run `pytest tests/test_schema.py -q` and observe the missing-module failure.
- [x] Implement dataclasses, validation, and JSON-safe conversion.
- [x] Run the focused test and then `pytest -q`.

### Task 2: Media metadata and shot segmentation

**Files:**
- Create: `src/football_tracking/video.py`
- Create: `tests/test_video.py`
- Create: `src/football_tracking/shot_config.py`

**Interfaces:**
- `VideoInfo.from_path(path)` returns dimensions, source FPS, duration, frame count, codec, and time base using OpenCV/FFprobe-compatible metadata.
- `ShotBoundary(frame_index, pts, reason, confidence)` is immutable.
- `detect_shots(frames, fps, manual_boundaries=())` returns ordered boundaries without duplicate indices.
- `iter_video_frames(path, start_frame=0, end_frame=None)` yields `(frame_index, pts, frame_bgr)` without loading the whole clip.

- [x] Write tests for deduplicated frame-712 manual boundaries, monotonic frame/timestamp output, and scene-threshold fallback.
- [x] Run focused tests and observe failure.
- [x] Implement metadata, streaming decode, and manual-boundary precedence.
- [x] Run focused tests and a read-only metadata check against `data/all-22-lions-rams-sample.mp4`.

### Task 3: Detector adapters and cached detections

**Files:**
- Create: `src/football_tracking/detector.py`
- Create: `tests/test_detector.py`
- Create: `src/football_tracking/cache.py`

**Interfaces:**
- `Detection(box_xyxy, score, class_name, class_id)` is immutable.
- `Detector.predict(frame_bgr) -> list[Detection]` converts BGR to RGB at the adapter boundary.
- `RFDETRDetector(model_size, checkpoint, device)` loads the pinned RF-DETR release lazily and validates class mapping.
- `DetectionCache.save/load` stores source hash, frame index, detector config hash, and detections.

- [x] Write a test that a sentinel RGB pixel reaches a fake predictor unchanged after BGR conversion, plus cache invalidation on config hash changes.
- [x] Run focused tests and observe failure.
- [x] Implement the protocol, RF-DETR lazy import, and JSONL cache.
- [x] Run focused tests; run a model import smoke test only if the package is installed.

### Task 4: Tracker adapters and short tracklets

**Files:**
- Create: `src/football_tracking/tracking.py`
- Create: `tests/test_tracking.py`

**Interfaces:**
- `TrackObservation(tracklet_id, frame_index, pts, bbox_xyxy_px, score, state)` is immutable.
- `TrackerAdapter.update(detections, frame_bgr, timestamp_s) -> list[TrackObservation]`.
- `RoboflowTracker(kind, fps, lost_track_buffer, enable_cmc)` wraps `trackers.ByteTrackTracker` or `trackers.BoTSORTTracker`, passes timestamps consistently, and resets per shot.
- `IoUTracker` is a deterministic test/smoke fallback; it is never presented as the production benchmark.

- [x] Write tests for empty-detection advancement, one-to-one crossing assignment, reset at shot boundaries, and timestamp propagation.
- [x] Run focused tests and observe failure.
- [x] Implement the fallback tracker and lazy Roboflow adapters.
- [x] Run focused tests and, if installed, a package signature smoke test for both adapters.

### Task 5: Team evidence and constrained identity resolution

**Files:**
- Create: `src/football_tracking/identity.py`
- Create: `tests/test_identity.py`

**Interfaces:**
- `TeamEvidence(team, score, source, crop_count)` stores aggregate evidence.
- `TrackletSummary(tracklet_id, team_evidence, observations, jersey_candidates)` is immutable.
- `resolve_teams(tracklets, field_mask) -> dict[tracklet_id, TeamEvidence]` uses torso color with an explicit unknown class.
- `match_tracklets(left, right, candidate_scores, threshold, margin) -> list[IdentityLink]` enforces one-to-one assignments and abstains below threshold/margin.
- `IdentityLink(left_key, right_key, decision, score, evidence_keys)` records accepted/rejected/insufficient evidence.

- [x] Write tests for team aggregation, officials/unknown handling, one-to-one matching, threshold abstention, and deterministic anonymous IDs.
- [x] Run focused tests and observe failure.
- [x] Implement color evidence, matching (SciPy when available, deterministic fallback otherwise), and stable ID generation.
- [x] Run focused tests and export a synthetic identity fixture.

### Task 6: Field calibration and trajectory projection

**Files:**
- Create: `src/football_tracking/calibration.py`
- Create: `tests/test_calibration.py`

**Interfaces:**
- `FieldPoint(x_yards, y_yards)` and `ImagePoint(x_px, y_px)` are immutable.
- `Homography.fit(image_points, field_points, reprojection_threshold_px)` validates non-collinearity and stores fit errors.
- `Homography.project(image_point) -> FieldPoint | None` maps bottom-center contact points.
- `project_observation(observation, homography, source)` writes field coordinates only when the homography is valid.

- [x] Write tests for known affine/projective mappings, degenerate landmarks, and null output for invalid calibration.
- [x] Run focused tests and observe failure.
- [x] Implement robust homography fitting using OpenCV and error reporting.
- [x] Run focused tests and a synthetic yard-grid round trip.

### Task 7: Exporters and instrumentation

**Files:**
- Create: `src/football_tracking/export.py`
- Create: `src/football_tracking/metrics.py`
- Create: `tests/test_export.py`

**Interfaces:**
- `write_observations_csv`, `write_observations_parquet`, `write_identities_json`, and `write_manifest_json` create deterministic artifacts.
- `render_annotated_video(input_path, output_path, observations, shots)` overlays boxes, team, anonymous IDs, and honest gaps while preserving source timing.
- `write_field_view(output_path, trajectories)` renders yard-space paths with shot/play provenance.
- `StageTimer` records decode, detection, tracking, identity, calibration, export, wall time, and memory where available.

- [x] Write tests for nullable schema columns, deterministic JSON ordering, and no trail across shot boundaries.
- [x] Run focused tests and observe failure.
- [x] Implement table/JSON/video/field exporters and timers.
- [x] Run focused tests plus a tiny generated-video render check.

### Task 8: CLI pipeline and local benchmark commands

**Files:**
- Create: `src/football_tracking/cli.py`
- Create: `src/football_tracking/__main__.py`
- Create: `scripts/benchmark_local.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- `python -m football_tracking run --input ... --output ... --model-size small --tracker botsort --manual-cut 712 --detector-checkpoint ...` runs the full local pipeline and writes the documented artifacts.
- `python -m football_tracking benchmark --input ... --output ...` runs decode, detector smoke/model, ByteTrack-vs-BoT-SORT, identity, and export benchmarks, recording unavailable gates explicitly.
- CLI exits nonzero for missing input/checkpoint in real mode and zero for a completed benchmark report with `status: incomplete` when optional model execution is unavailable.

- [x] Write CLI argument, missing-input, and benchmark-report tests.
- [x] Run focused tests and observe failure.
- [x] Implement command routing, configuration hashing, shot reset, cache reuse, and artifact manifest.
- [x] Run focused tests and `python -m football_tracking --help`.

### Task 9: Dependencies, real clip run, and verification

**Files:**
- Modify: `pyproject.toml`
- Create: `requirements-lock.txt` only after successful installation resolution
- Create: `artifacts/<run-id>/` generated by the benchmark (ignored if appropriate)
- Modify: `README.md` with actual run commands and measured status

- [x] Install or resolve RF-DETR 1.10.1, trackers 2.6.0, supervision 0.30.2, OpenCV, SciPy, and pytest in the local environment; record exact versions.
- [x] Run the complete unit suite and static import checks.
- [x] Run the benchmark on the entire supplied clip, preserving all source frames and reporting model/weight status.
- [x] Run ByteTrack and BoT-SORT with the same generic RF-DETR detection cache, export annotated video/CSV/JSON/metrics, and measure total/per-stage wall time and output frame count.
- [x] Inspect generated artifacts with FFprobe and a contact sheet; verify the frame-712 cut, no raw track state or pixel trail carried across it, no replay double-counting, and no fabricated off-screen players in the renderer.
- [x] Run privacy/path scan, `git diff --check`, and final requirement-by-requirement audit. Report achieved metrics separately from proposed gates and leave the football-fine-tuned accuracy gate explicit.
