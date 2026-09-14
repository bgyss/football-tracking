# McByte Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, evidence-gated McByte tracking and evaluation lane without changing the BoT-SORT default.

**Architecture:** Keep the core pipeline independent of mask packages. A dedicated lazy McByte adapter owns RGB conversion, explicit model paths/device configuration, and effective-mask telemetry. Strict cache metadata makes detector output replayable across tracker variants; a separate evaluator consumes only reviewed references and otherwise reports `not_evaluated`.

**Tech Stack:** Python 3.11, NumPy, OpenCV, RF-DETR 1.10.1, trackers 2.6.0, supervision 0.30.2, optional `trackers[mask]`, optional TrackEval, pytest.

**Spec:** `docs/goals/mcbyte-integration.md`, `docs/mcbyte-evaluation-plan.md`, and `docs/research/mcbyte.md`.

## Global Constraints

- Keep BoT-SORT as the default and retain the proxy benchmark as a separately labeled smoke lane.
- Feed native-rate frames, preserve source frame index/PTS, use RGB only at the McByte boundary, and reset state at every shot.
- Do not fetch model weights implicitly; requested mask-enabled runs require explicit existing files and report hashes/device/effective-mask state.
- Reject shared caches with mismatched source/checkpoint/config/proxy provenance, duplicate/missing/out-of-range frames, or non-explicit empty frames.
- Score only reviewed references, use one-based evaluator frame numbers behind a reversible mapping, and report absent labels as `not_evaluated`.
- Keep media, weights, raw references, and generated comparison artifacts ignored; do not commit, upload, train, or change defaults.

---

### Task 1: McByte dependency contract and adapter

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `src/football_tracking/tracking.py`, `src/football_tracking/cli.py`
- Test: `tests/test_tracking.py`, `tests/test_cli.py`

**Interfaces:** `McByteTracker(fps, device, sam_checkpoint, cutie_checkpoint, enable_masks, shot_id)` implements `TrackerAdapter`; `effective_mode()` emits requested/effective mask state, device, asset hashes, and fallback reason.

- [ ] Write failing tests for BGR-to-contiguous-RGB conversion exactly once, explicit missing weights/device errors, empty-frame advancement, reset, and runtime mask fallback reporting.
- [ ] Run `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_tracking.py tests/test_cli.py -q` and confirm the named tests fail.
- [ ] Add `mcbyte` extra pinned to `trackers[mask]==2.6.0`; resolve `uv.lock`; inspect installed 2.6.0 signatures and import behavior.
- [ ] Implement a lazy adapter that passes only the verified constructor/update arguments, creates no implicit downloads, preserves frame/PTS and shot namespaces, and exposes effective mode.
- [ ] Add CLI flags for mask mode, explicit SAM/Cutie paths and device; ensure `--tracker mcbyte` never alters the default.
- [ ] Re-run focused tests and the optional-extra import/preflight smoke command.

### Task 2: Strict shared cache replay

**Files:**
- Modify: `src/football_tracking/cache.py`, `src/football_tracking/cli.py`, `src/football_tracking/metrics.py`
- Test: `tests/test_detector.py`, `tests/test_cli.py`

**Interfaces:** `DetectionCache.load_strict(..., frame_count, provenance)` returns complete per-frame detections or raises `CacheMismatch`; cache metadata includes source hash, checkpoint content hash, class mapping, detector config, and proxy provenance.

- [ ] Write failing tests for wrong source/checkpoint, proxy mismatch, missing/duplicate/out-of-range indices, and explicit empty-frame acceptance.
- [ ] Run the focused cache tests and verify each fails for the intended validation reason.
- [ ] Add strict metadata/versioning and checkpoint-content hashing; preserve the legacy local-cache behavior only where safe.
- [ ] Add `--detection-cache` as an explicit controlled-replay input; reject invalid input rather than regenerating detections.
- [ ] Re-run cache and CLI tests.

### Task 3: Reviewed-reference importer and evaluator

**Files:**
- Create: `src/football_tracking/evaluation.py`
- Modify: `pyproject.toml`, `src/football_tracking/cli.py`
- Test: `tests/test_evaluation.py`

**Interfaces:** `load_reviewed_mot_reference(path)` rejects unreviewed/ambiguous records; `evaluate_tracking(predictions, reference, config)` produces per-shot/aggregate ID metrics or `not_evaluated`.

- [ ] Write analytically known perfect/swap/gap tests for frame-number conversion, ignored/unlabeled frames, namespaces, coverage, switches, and fragmentation.
- [ ] Run the evaluator tests and confirm missing-module failures.
- [ ] Add an optional pinned TrackEval dependency plus a deterministic importer/config wrapper; reject absent or unreviewed data rather than treating it as empty truth.
- [ ] Add real-cache comparison reporting for ByteTrack, BoT-SORT CMC on/off, McByte masks on/off, distinguishing unsupported ablations and degraded masks.
- [ ] Re-run evaluator plus full unit suite.

### Task 4: Evidence artifacts and documentation

**Files:**
- Modify: `README.md`, `docs/research/mcbyte.md`, `docs/mcbyte-evaluation-plan.md`, `.gitignore`
- Create: `docs/evidence/mcbyte-evaluation.md`
- Test: `tests/test_cli.py`

- [ ] Write failing CLI/report tests for explicit integration/real-backend/quality/held-out statuses and tracker-setting-specific run identities.
- [ ] Implement machine-readable comparison manifests containing timing, peak memory where available, device, effective masks, provenance, evaluator settings, and artifact locations.
- [ ] Document exact optional install/import/preflight/replay commands, known sources/licenses/hashes, unavailable prerequisites, and evidence distinctions.
- [ ] Run the prescribed suite/help/diff checks, then inspect any authorized real assets. Only run real mask-active/full comparison if source/cache/weights/reviewed references and supported device are present; otherwise record exact missing gates.

