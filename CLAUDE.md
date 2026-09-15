# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from the project root with `uv`, pinned to the in-repo cache:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev                      # core + pytest
UV_CACHE_DIR=.uv-cache uv sync --extra dev --extra roboflow     # + RF-DETR/trackers/supervision
UV_CACHE_DIR=.uv-cache uv sync --extra dev --extra roboflow --extra mcbyte --extra evaluation

UV_CACHE_DIR=.uv-cache uv run pytest -q                         # full suite
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_tracking.py -q  # one file
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_cli.py::test_name -q

UV_CACHE_DIR=.uv-cache uv run python -m football_tracking --help
```

The test suite runs without the heavy extras: adapters import lazily and
`tests/test_evaluation.py` uses `pytest.importorskip("trackeval")`. Keep it that way —
never make `rfdetr`, `trackers`, `supervision`, or `trackeval` a core dependency.

Fast end-to-end smoke run with no model weights (results are explicitly labeled proxy):

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run \
  --input data/all-22-lions-rams-sample.mp4 --output artifacts/smoke \
  --detector synthetic --tracker iou --manual-cut 712
```

The sample clip is 1,424 frames at 60000/1001 fps with a manually verified cut at frame
712; pass `--manual-cut 712` to skip scene-detection when working with it.

## Architecture

Offline, two-pass pipeline in `src/football_tracking/`, driven entirely by `cli.py`
(`run`, `batch`, `benchmark`, `cache`, `merge-cache`). `run_pipeline` is the spine: decode →
shot boundaries → detect (or cache hit) → per-shot track → team/identity → calibrate →
export → evaluate → manifest.

- `video.py` — `VideoInfo` (ffprobe metadata, `time_base`, real PTS list) and
  `iter_video_frames`. Every frame is iterated at native rate; nothing resamples.
- `detector.py` — `Detector` protocol, lazy `RFDETRDetector`, and `SyntheticDetector`
  (`is_proxy=True`, propagated into manifests and cache provenance).
- `cache.py` — JSONL detection cache. `load()` is the ordinary per-run cache;
  `load_strict()` / `merge_strict_parts()` back controlled comparisons and reject on any
  mismatch of source hash, detector config hash, checkpoint hash, class mapping, proxy
  flag, or frame coverage. Detector cache identity deliberately excludes tracker
  settings; tracker settings go into run identity instead.
- `tracking.py` — `TrackerAdapter` protocol with `IoUTracker` (dependency-free
  fallback), `RoboflowTracker` (botsort/bytetrack), and `McByteTracker` (opt-in masks
  via SAM + Cutie, explicit device, `effective_mode()` reports runtime mask fallback).
  A fresh tracker is constructed per shot; tracklet ids are shot-local (`shot-N:tID`).
- `identity.py` — conservative team evidence, globally ambiguous tracklet matching, and
  union-find `stable_anonymous_ids` with same-shot component guards.
- `identity_resolution.py` — play-scoped all-view candidate resolution with explicit
  partial and abstained outcomes.
- `tracklet_refinement.py` — reviewed, immutable within-shot split overlays passed by
  `--reviewed-splits`; raw tracker rows remain unchanged.
- `replay.py` — reviewed play-time alignment (`--play-alignment`, refuses anything not
  explicitly marked `reviewed: true`), PTS-based maps, and auditable cross-shot candidate
  evidence from view-invariant signals only (team agreement, calibrated field position at
  aligned play time, trajectory shape, overlap span, and uncertainty). Cross-shot resolution runs when `--play-alignment`
  supplies reviewed snap anchors and requires genuinely shot-specific calibration for
  the two shots being resolved (a shared `"*"` homography does not qualify, since
  applying one camera pose's transform to another shot would make "field position" a
  restatement of image coordinates); otherwise it abstains. Every decision — resolved,
  abstained, or not attempted — is recorded in `identity-links.json`.
- `calibration.py` — homography fit from landmark JSON with pixel-space RANSAC and
  independent yard residuals; without `--calibration` the yard columns stay empty and only
  image-space positions are exported.
- `calibration_timeline.py` — reviewed PTS-scoped fits, withheld-landmark eligibility, and
  field-only motion propagation proposals.
- `annotations.py` / `field.py` — source-hashed reviewed annotation manifests and the
  canonical NFL field coordinate template.
- `scripts/fit_calibration_timeline.py` — convert a reviewed annotation manifest into the
  schema-v2 PTS-scoped calibration artifact.
- `scripts/build_annotation_manifest.py` — create an unreviewed manifest template from the
  extracted full-game review pack.
- `scripts/build_play_inventory.py` — scan long recordings into unreviewed candidate shot
  intervals for play grouping and calibration review.
- `evaluation.py` — reviewed MOT-style reference import and HOTA/IDF1-style metrics.
  Without `--reviewed-reference`, `tracking-evaluation.json` says `not_evaluated`.
- `memory.py` — `MemoryBudget`, a peak-RSS guard (default 2048 MiB) sampled at stage
  boundaries; on macOS the hard limit cannot be enforced before McByte mask
  construction, which is why `--allow-unbounded-memory` exists.
- `run --start-frame/--end-frame` — bounded source windows for long recordings; window
  scope is recorded and partial detector caches are never assumed to cover other windows.
  Bounded CFR windows use a declared PTS/frame step; full passes retain ffprobe PTS.
  Multi-play alignment manifests require explicit `--play-id` selection.
- `batch` — runs each reviewed alignment-set play with declared `shot_ranges` in its own
  bounded output directory and emits `batch.json`; it reports `complete_with_unresolved`
  when execution finished but a child identity/evaluation gate remains open, and only
  reports `complete` when every child promotion gate passes.
- `schema.py` — `Observation` / `RunManifest` frozen dataclasses with validating
  `__post_init__`. These are the stable contract; `export.py` writers follow it.

Each `run` writes a fixed artifact set to `--output`: `annotated.mp4`,
`observations.csv`/`.parquet`, `trajectories.csv`, `play-trajectories.csv`, `identities.json`, `field-view.png`,
`shots.json`, `calibration.json`, `calibration-quality.json`, `identity-links.json`,
`analysis-config.json`, `review.json`, `tracklet-refinement.json`, `tracking-evaluation.json`, `metrics.json`, `artifact-validation.json`, `run-manifest.json`,
`detections.jsonl`.

## Project conventions

- **Never let output imply evidence it does not have.** Proxy detectors, generic
  (non-football-finetuned) checkpoints, missing ground truth, unavailable backends, and
  degraded mask runs are all recorded as explicit status strings (`proxy`,
  `not_evaluated`, `unavailable`, `unresolved_cross_view`, `complete_with_unresolved`)
  rather than omitted or scored. Benchmarks report a failed or missing gate as such;
  never substitute IoU/synthetic results for a requested backend.
- BoT-SORT is the default tracker. McByte and everything else is opt-in; do not change
  the default without the evaluation gates in `docs/evaluation-plan.md`.
- Shots are tracked independently. The second shot of the sample is a replay from the
  end zone — never append it to the first shot's trajectory as a continuation.
- Outputs are deterministic and sorted (see `_json_safe`, the JSON writers); keep them
  diffable.
- `artifacts/`, `data/*.mp4`, and model weights are gitignored. Keep footage,
  checkpoints, and bulky generated output out of Git.
- New model/weight acquisition is never implicit: checkpoints are passed by path and
  hashed into the manifest; `TORCH_HOME` is redirected into `artifacts/` for Cutie.

## Docs

`docs/system-design.md` (architecture and rationale), `docs/evaluation-plan.md`
(acceptance gates), `docs/mcbyte-evaluation-plan.md` and `docs/goals/mcbyte-integration.md`
(the McByte contract, including scope boundaries), `docs/accuracy-calibration-replay.md`,
`docs/annotation-and-data-generation.md` (CVAT/SAM2/FiftyOne labeling loop),
`docs/evidence/` (verified measurements — extend rather than restate).
