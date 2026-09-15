# Calibration and Cross-Shot Identity Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. Do not commit, acquire model weights, or launch training unless requested.

**Goal:** Demonstrate correct anonymous identity across reviewed views of the same play, with valid yard coordinates, honest evaluation, and explicit abstention on insufficient evidence.

**Architecture:** Repair measurement and matching safety first, then introduce reviewed, frame-scoped calibration and PTS-based play alignment. Build clean within-shot identity segments and combine them through a constrained play-level identity graph; retain all original observations and evidence. Use manual field annotations and local field-motion propagation before evaluating learned calibration or appearance models.

**Tech Stack:** Python 3.11+, NumPy, OpenCV, SciPy, PyArrow, pytest, uv; pinned TrackEval remains optional. FFmpeg/ffprobe for local media indexing and previews.

**Spec:** [Current research](../../evidence/cross-shot-identity-research-2026-09-15.md), [dated repository audit](../../evidence/cross-shot-identity.md#2026-09-15-correctness-audit), and the existing [acceptance gates](../../evaluation-plan.md). This continues the [previous implementation plan](2026-09-14-cross-shot-identity.md); its implemented plumbing is retained, while its static calibration and first-two-shot limitations are addressed here.

## Decision and scope

Calibration is necessary but insufficient. Simply supplying two landmark files would enable a resolver with reproducible ambiguity and evaluation defects. Fix those defects before interpreting accepted links as successful identity resolution.

Deliver this in three milestones:

1. **Safe and measurable:** tasks 1–3; bad or ambiguous evidence cannot create a success claim.
2. **One play verified:** tasks 4–7; reviewed sample views meet geometry, timing, and identity gates on real detections.
3. **Within-game validation:** tasks 8–9; independent plays from the full game pass frozen evaluation. Different-game generalization remains a separate release gate.

Identity in this plan is anonymous and play-scoped. A persistent identity across unrelated plays, or a roster name, requires additional reviewed jersey/roster evidence. Repeated formations and matching uniform colors are insufficient to link across plays.

## Global constraints

- BoT-SORT stays the default; preserve per-cut tracker resets and strict cache provenance.
- No model weights, cloud inference, external uploads, or training are needed for the initial fixes or annotation export.
- Core imports and tests work without RF-DETR, trackers, supervision, or TrackEval.
- Preserve source frame index, PTS, and time base; clip-local frames require a reversible source mapping.
- Keep `shot_id`, `play_id`, and `player_id` distinct. Preserve raw tracks when correcting identity segments.
- No cross-camera image-coordinate matching or shared `"*"` calibration fallback for identity.
- Human review is required for acceptance labels; generated proposals remain `reviewed: false`.
- Preserve zero observed false merges and >= 0.80 human-resolvable shared-player coverage; never lower gates after seeing results.
- Store footage, crops, detection caches, and bulky generated output under ignored data/artifact paths. Version small schemas, policies, split IDs, hashes, and evidence summaries.
- No-alignment identity output remains unchanged unless a separately enabled within-shot correction mode is requested.

## Verified starting point

Audit baseline: commit `156f751`. The current implementation has shot-local tracking, reviewed snap-anchor loading, static shot homographies, a geometric candidate score, Hungarian assignment, anonymous union-find IDs, and cross-shot reports. It has no demonstrated real-footage acceptance result in the committed evidence.

| Priority | Finding | Evidence / consequence |
| --- | --- | --- |
| P0 | Calibration units are inconsistent | `Homography.fit` fits pixels to yards, then labels destination residuals and RANSAC threshold `_px`. A threshold of 3 is applied in yards. |
| P0 | Ambiguous assignments can be accepted | `match_tracklets` excludes already assigned columns from alternatives; a 2-by-2 all-0.9 matrix yields two `same` links. |
| P0 | Coverage denominator depends on predictions | Missing shared reference players disappear; a fixture detecting only one of two shared players reports coverage 1.0. |
| P0 | Top-level IDF1 is a detection F1 calculation | A fixture with four identity switches reports aggregate IDF1 1.0. Standard metrics exist only in the optional per-shot TrackEval block. |
| P0 | Partial matching is reported as resolved | One accepted link sets `resolved_cross_view` and run `complete`; omitted or uncalibrated aligned shots can be absent from `unresolved_shots`. |
| P1 | Calibration has no temporal support or held-out validation | One fit is reused for a whole shot; export labels any nonempty calibration mapping `valid`. |
| P1 | Timing assumes constant rate and one play | `(frame-anchor)/fps` ignores PTS and playback changes; no anchor-to-shot containment or event correspondence validation. |
| P1 | Raw tracklet is treated as a pure identity | Mixed tracks and nonoverlapping fragments cannot be handled safely by a simple one-to-one tracklet match. |
| P1 | Weak evidence lacks eligibility gates | Team labels are forced for usable crops; five samples need span only about 67 ms at 60 fps; no contact/geometry uncertainty enters scoring. |
| P1 | Reproduction identity omits inputs | Run/config hashes omit calibration, alignment, prototypes and resolver policy; different identity results can share a run identity. |
| P2 | Full-game execution is not bounded by play | Single-play alignment, first-two-shot resolution, retained tracks/crops/cache rows, and per-pair sample searches need scope and performance work before a full-game run. |

## Full-game asset and data strategy

Read-only inspection found `data/all-22-lions-rams.mp4` in the main checkout, not this linked worktree. Use an explicit input path or create an ignored local data link during implementation; do not duplicate the file. ffprobe reports 1920×1080, 306,151 video frames, video duration 5107.618915 seconds (85m 7.62s), frame rate 19001/317 and time base 1/19001. These are container metadata, not a full decode/count audit.

Twelve seek previews at 30/35/40, 600/605/610, 1800/1805/1810, and 3600/3605/3610 seconds show both camera types, usable field markings, and changing framing. These are scouting timestamps, not reviewed anchors or exact frame annotations. Do not copy the sample clip's frame-712 boundary into the full game; locate and verify the sample's source interval independently.

Proposed initial labeling budget, adjusted for actual evaluability before running models:

- Annotate the existing two-shot sample densely at the declared approximately 10 Hz evaluation cadence; inspect contacts/cuts at native cadence.
- Select 20 additional paired plays across the game: 12 development and 8 sealed within-game test plays. Keep both views and nearby repetitions of a play together. If fewer qualify, report the shortfall rather than silently substituting easy plays.
- Start with three geometry keyframes per shot (early, middle, late): 40 shots × 3 = 120 keyframes. Add frames at pan/zoom changes and loss of line support.
- At each usable keyframe, target 8–12 well-distributed fitting landmarks and at least 4 separately labeled withheld landmarks. Mark insufficient support instead of fabricating invisible points.
- Annotate snap plus a later corresponding event for time-map fitting, and at least one additional corresponding event for validation; include ambiguous and negative pairings.
- Label visible players, team/role, track identity, visibility, truncation, ground-contact confidence, and human-resolvable cross-view identity. Label fully hidden intervals explicitly.
- Separate calibration fitting points, withheld geometry labels, and player identity truth. Neither resolver output nor fitted geometry may manufacture identity ground truth.

The same-game test is useful engineering evidence, but cannot satisfy the existing release requirement of at least ten additional plays from at least three games with disjoint training/development and held-out games.

## Implementation tasks

### Task 1: Correct the evaluation contract

**Modify:** `src/football_tracking/evaluation.py`, `tests/test_evaluation.py`, `docs/evaluation-plan.md`.

**Interface:** Keep `evaluate_tracking` and `evaluate_cross_shot_identity`; version their report schema. Separate `diagnostics`, `standard_metrics`, `shared_player_coverage`, and `component_errors`. Standard metrics carry `evaluated` or `unavailable` plus evaluator version.

- [ ] Add a regression fixture with two shared reference players, detections for only one, and one correct link. Assert shared-player coverage is 0.5, not 1.0.
- [ ] Add a fixture with perfect boxes but alternating identities. Assert standard IDF1 falls below 1.0; without TrackEval assert it is unavailable, not replaced by detection F1.
- [ ] Add fixtures for mixed tracklets, duplicate predictions matching one reference object, absent identity-map entries, unknown truth, and fragmentation-dependent pair counts.
- [ ] Replace independent best-IoU voting with injective per-frame reference association. Retain per-tracklet truth distributions and mixed/unknown attribution rather than majority identity alone.
- [ ] Define the primary coverage denominator from reviewed shared identities per play/view pair, independent of detected tracklets. Report raw numerator/denominator, player-pair and tracklet-pair diagnostics separately. Fragmenting a player must not improve the headline score.
- [ ] Count accepted but unattributable links separately; they cannot establish a precision/zero-false-merge gate. Report component contamination and any simultaneous identity collision. Require adequate reviewed coverage before declaring a pass.
- [ ] Rename approximate HOTA/AssA/IDF1 outputs to explicit diagnostics; aggregate real standard metrics using TrackEval's supported sequence-combination API. Do not average percentages as a substitute for that API.

Regression contract:

```python
assert report["shared_player_coverage"] == {"correct": 1, "eligible": 2, "value": 0.5}
assert report["gate"]["coverage_at_least_0_80"] is False
assert report_without_trackeval["standard_metrics"]["status"] == "unavailable"
```

**Verify:** `UV_CACHE_DIR=.uv-cache uv run --extra dev python -m pytest tests/test_evaluation.py -q`; then repeat with `--extra evaluation` for the standard-metric fixtures. The optional-dependency test must run in the latter check, not skip.

### Task 2: Make assignment ambiguity and component conflicts hard gates

**Modify:** `src/football_tracking/identity.py`, `tests/test_identity.py`.

**Interface:** Retain `match_tracklets`; add explicit unmatched choices and component validation before `stable_anonymous_ids`. Return evidence for the selected assignment, alternatives, and conflict reasons.

- [ ] Add and run this failing regression before changing matching:

```python
scores = {(a, b): 0.9 for a in ("a", "b") for b in ("c", "d")}
links = match_tracklets(["a", "b"], ["c", "d"], scores)
assert all(link.decision == "insufficient_evidence" for link in links)
```

- [ ] Add rectangular-matrix, missing-edge, row/column competitor, equal-optimum, and below-threshold fixtures; verify input ordering does not affect results.
- [ ] Solve with explicit dummy unmatched columns. Remove ineligible edges before assignment. For each proposed real edge, forbid it and recompute the best global objective; require a frozen global ambiguity margin, and conservative row/column checks. Assigned competitors must remain alternatives in the ambiguity analysis.
- [ ] Use the core SciPy dependency consistently; avoid silently changing to greedy behavior on an import failure.
- [ ] Before union, reject components containing incompatible plays, contradictory strong team/jersey evidence, or distinct segments overlapping in the same shot/time. Preserve the reason and original IDs.
- [ ] Keep `stable_anonymous_ids` deterministic; make its inputs explicitly validated links. Test a three-view transitive conflict and a valid three-view component.

**Verify:** `UV_CACHE_DIR=.uv-cache uv run --extra dev python -m pytest tests/test_identity.py -q`.

### Task 3: Correct units, reporting, and reproducibility

**Modify:** `calibration.py`, `cli.py`, `export.py`, `schema.py` under `src/football_tracking/`; corresponding existing tests.

**Interface:** Introduce schema-versioned calibration diagnostics with `fit_error_px`, `heldout_error_yards`, `status`, and `reason`; preserve the image-to-field matrix used by callers.

- [ ] Add a noisy-landmark regression that distinguishes pixel from yard residuals and checks scale changes. Add finite-input, repeated-point, near-collinear, singular-matrix, and projection-at-horizon cases.
- [ ] Fit field-to-image with OpenCV RANSAC so `reprojection_threshold_px` actually uses pixel units; validate and invert the fit for image-to-field projection. Compute residuals in both directions with explicit units. Do not merely rename existing erroneous pixel values.
- [ ] Separate fit inliers/outliers from withheld labels. A four-point exact fit may be mathematically usable but is `unvalidated` for identity until independent checks pass.
- [ ] Reject nonfinite projections and near-zero homogeneous denominators explicitly. Bound valid support spatially; do not clamp impossible coordinates into the field.
- [ ] Migrate legacy calibration explicitly: expose version and historical units, require review/refit for identity eligibility, and never silently reinterpret old thresholds.
- [ ] Derive export validity from actual calibrator status; use `not_provided`, `unvalidated`, `valid`, `partial`, or `invalid` with counts and reasons.
- [ ] Separate execution completion from identity resolution and evaluation gate state. Any unmatched eligible track, missing calibration, or skipped aligned shot keeps a partial/unresolved status. A lack of reference gives `not_evaluated`, even when links are accepted.
- [ ] Hash source, detector/cache provenance, shot boundaries, calibration file, alignment file, prototype file, resolver policy, contact policy, and schema/code versions into analysis identity. Keep evaluation-reference provenance separate and recorded. Derive proxy status from cache provenance on cache hits as well as live detectors.

**Verify:** existing calibration/export/schema/CLI suites, plus a cache-hit synthetic run that retains proxy status and two runs differing only in calibration that receive different analysis IDs.

### Task 4: Build the reviewed annotation and field-template contract

**Create:** `docs/annotation-schema.md`, `src/football_tracking/annotations.py`, `src/football_tracking/field.py`, `scripts/build_identity_review_pack.py`, `tests/test_annotations.py`, `tests/test_field.py`.
**Modify:** `docs/annotation-and-data-generation.md`.

**Interfaces:** `load_annotation_manifest(path, source_sha256)` validates a versioned manifest. `field_landmark(landmark_id)` returns the canonical field coordinate. The review-pack script emits original-resolution frames, crop proposals, source mappings and unreviewed annotation records; no auto-review behavior.

- [ ] Test source-hash mismatch, out-of-shot frame, duplicate IDs, conflicting play splits, cropped/resized coordinate conversion, and unreviewed input rejection.
- [ ] Freeze the NFL template: x 0–120 yards including end zones; y 0–160/3 yards; goal lines x=10 and x=110; hash rows y=23.583333… and y=29.75. Name which physical end and sideline define the origin. Validate against the official field rules linked in the research note.
- [ ] Use semantic landmark IDs including field half and line identity. Require global orientation evidence to disambiguate repeated yard numbers and mirrored layouts. Exclude goalposts, people and painted-number centroids from ground-line intersection labels.
- [ ] Include source hash, source frame, PTS/time base, shot interval, camera label, play ID, split, image dimensions, fit/withheld role, annotation confidence, reviewer, revision, and review timestamp.
- [ ] Export the sample review pack first, then the proposed 20-play inventory. Contact-sheet seeks are discovery aids only; final extraction must resolve exact frames and PTS.
- [ ] Use the existing CVAT workflow for human corrections. Retain proposals and reviewed overlays separately; stop acceptance evaluation for unreviewed labels while continuing code work.

**Verify:** annotation and field tests; inspect a generated frame against the original video and round-trip a reviewed box/landmark through full-resolution coordinates.

### Task 5: Add frame-scoped calibration and motion propagation

**Create:** `src/football_tracking/calibration_timeline.py`, `tests/test_calibration_timeline.py`.
**Modify:** `calibration.py`, `cli.py`, `schema.py`, `export.py` and their tests.

**Interfaces:** `CalibrationTimeline.at(shot_id, pts) -> CalibrationEstimate | None`. `CalibrationEstimate` contains image-to-field H, calibration ID, support interval/polygon, source keyframe IDs, fit diagnostics, and validity reason. Observation projection consumes this estimate and preserves invalid reasons.

- [ ] Test known pan/zoom sequences, a cut, absent calibration, long propagation gaps, field support leaving frame, mirrored landmarks, and the case where fitting points look good but withheld points fail.
- [ ] Load reviewed time-keyed fits and evaluate withheld points independently. A static fit may cover a full shot only after early/middle/late validation supports that interval.
- [ ] Add deterministic field-only motion propagation as the first automatic baseline: mask players/officials/graphics; estimate robust image transforms from static field support; re-anchor to absolute landmarks regularly.
- [ ] For a transform G mapping keyframe pixels to current pixels, compose `H_current = H_keyframe @ inv(G)`. Verify this direction with a synthetic pan regression.
- [ ] Stop propagation at cuts, inadequate spatial support, validation failure, or a frozen maximum gap. Select the maximum gap on development data, record it in policy, then freeze it before held-out runs.
- [ ] Do not interpolate matrix entries. If interpolation is needed, interpolate validated landmark trajectories and refit, or add a separately evaluated physical camera model; never extrapolate through unsupported motion.
- [ ] Attach position uncertainty from withheld geometry residuals, local projection sensitivity, and contact quality. Initially invalidate truncated/hidden feet and pile contacts; retain bottom-center as an explicitly approximate source where evaluable.
- [ ] Report landmark median/p95 in yards and independently reviewed player-contact error. Gate each view and motion stratum, not only a pooled mean. Require median <=1 yard, p95 <=2 yards, and >=90% valid evaluable contact coverage under the predefined policy.

**Verify:** timeline/calibration tests and real reviewed sample overlays at early/middle/late times. If a static baseline passes these gates, retain it for that interval; learned calibration is unnecessary there.

### Task 6: Align play time using source timestamps

**Modify:** `src/football_tracking/replay.py`, `cli.py`, `schema.py`, `export.py`; `tests/test_replay.py`, `tests/test_cli.py`.

**Interfaces:** Add `PlayTimeMap.at(shot_id, pts) -> float | None`, fitted from reviewed event correspondences and restricted to valid intervals. Alignment manifest contains multiple plays, each with shots, anchors, mapped play times, and held-out check events.

- [ ] Test variable PTS, a rate change, a freeze, duplicate event labels, mismatched events, anchors outside shots, unknown shots, nonmonotonic mappings, and missing temporal overlap.
- [ ] Preserve media time and map it separately to play time. With two corresponding events fit an affine map; use reviewed piecewise segments for edits. A one-anchor legacy map remains an explicitly unvalidated equal-rate assumption.
- [ ] Validate on a separate corresponding event and record residual milliseconds. Start with a proposed 50 ms maximum residual; on fast motion require the timing-induced displacement to fit within the geometric uncertainty budget. Freeze the rule before test evaluation.
- [ ] Pair samples on a monotonic common play-time grid with bounded interpolation and no reuse across gaps. Require overlap duration and distinct temporal support, not only five adjacent frames. Proposed initial duration is 0.5 seconds, tuned on development only.
- [ ] Export `play_time_s`, time-map ID and timing quality alongside unchanged PTS/time base; never append replay duration to live action.

**Verify:** replay/CLI tests and synchronization overlays for the reviewed sample. If correspondence is uncertain, report unaligned and abstain.

### Task 7: Repair local identity and team eligibility before fusion

**Create:** `src/football_tracking/tracklet_refinement.py`, `tests/test_tracklet_refinement.py`.
**Modify:** `identity.py`, `cli.py`, `schema.py`, existing team tests.

**Interfaces:** `refine_tracklets(raw_tracks, reviewed_splits, policy)` emits immutable identity segments with source-track references and valid intervals. Begin with reviewed split overlays; automatic purity detection only proposes review. Team evidence adds assignment coverage, ambiguity, and role eligibility.

- [ ] Test an identity-swapped raw track, two nonoverlapping fragments of one player, two overlapping teammates, official crops, conflicting crop votes, and unknown teams.
- [ ] Apply reviewed split corrections without mutating raw observations or detection caches. Track raw versus refined evaluation independently.
- [ ] Aggregate torso evidence over quality-selected, temporally spread crops; add a rejection threshold and winning-versus-runner-up evidence. Explicitly handle officials and sideline personnel; nonempty crops must not force a player-team identity.
- [ ] Use reviewed team prototypes initially to anchor labels across views. Team equality is a compatibility constraint, not teammate identity evidence.
- [ ] Join local fragments only with nonoverlapping visibility and strong continuity evidence; retain abstention otherwise. Do not use ground-truth identities as production features.
- [ ] Run oracle-local-track, oracle-geometry, and oracle-timing ablations independently to attribute remaining failures. Mark them diagnostic upper bounds.

**Verify:** refinement/identity tests; score within-shot IDF1 >=0.90 and <=1 switch per shot on the sample using standard metrics; report team accuracy >=0.98 with >=0.95 assignment coverage. If detection recall fails, start the existing targeted detector-data loop rather than trying to hide misses through linking.

### Task 8: Resolve every eligible view within each play and scale by play

**Create:** `src/football_tracking/identity_resolution.py`, `tests/test_identity_resolution.py`.
**Modify:** `replay.py`, `cli.py`, `export.py`, `schema.py` and integration tests.

**Interfaces:** `resolve_play_identities(play, segments, calibration, time_map, teams, policy)` returns validated links plus all rejection/skipping evidence. `cli.py` orchestrates play batches; detection cache identity remains tracker-independent.

- [ ] Test a three-view play, a missing calibration in the middle view, distinct plays with identical formations, fragmented players, transitive conflicts, and input-order determinism.
- [ ] Build candidates from valid synchronized field samples. Use robust distance distributions, overlap duration, local uncertainty and trajectory evidence; do not treat a fixed six-yard gate or a weighted score as a probability.
- [ ] Apply Task 2's assignment and component checks to compatible view pairs. Allow multiple nonoverlapping segments of the same player within one view through validated components, rather than forcing one raw tracklet per player per shot.
- [ ] Record per-candidate sample count/span, position residual, shape evidence, team quality, timing/calibration IDs, alternatives, ambiguity margin, and rejection reason. Account for every aligned shot and segment, including those with no valid coordinates.
- [ ] Namespace anonymous IDs by play and version. Keep replay observations distinct; if a canonical trajectory is produced, select/fuse one position per play-time bin with provenance and uncertainty so distance/participation is not doubled.
- [ ] Process one play/window at a time and spill raw observations/crops/cache chunks. Index time samples for matching rather than repeatedly searching full-shot arrays. Measure RSS and runtime on the 20-play subset before attempting 306,151 frames.
- [ ] Preserve default no-alignment behavior and shot-local trail breaks, including invalid gaps within a shot. Report `partially_resolved` for partial results and a separate evaluated gate result.

**Verify:** resolver/CLI/export tests, two-play replay deduplication fixture, bounded-memory subset run, and full reviewed sample acceptance report.

### Task 9: Freeze policy, evaluate independently, and decide optional models

**Modify:** `docs/evaluation-plan.md`, `docs/evidence/cross-shot-identity.md`, `docs/accuracy-calibration-replay.md`, `README.md`.
**Create:** dated ignored run artifacts and a small versioned acceptance summary.

- [ ] Freeze source/split hashes, reviewed annotation revisions, thresholds, model/cache versions, calibration/contact/time policies and evaluator version before the sealed eight-play test.
- [ ] Compare on identical real detections: no joins; reviewed static geometry; time-varying geometry; validated timing; refined local tracks/teams; final constrained resolver. Change one layer at a time.
- [ ] Publish correct/false/unknown link counts, independent shared-player coverage, component contamination, per-view geometry/contact errors, timing residuals, within-shot standard metrics, team coverage, uncertainty, runtime and memory.
- [ ] Require zero observed false merges and >=0.80 reviewed shared-player coverage on the sample and report each test play separately. Any unreviewed accepted link or unmet prerequisite prevents a pass. Publish denominators; zero observed errors is not a universal guarantee.
- [ ] Keep an error queue with exact source frames for every false merge, missed shared player, invalid calibration interval and mixed local segment.
- [ ] Only if geometry/timing/local purity pass but identity coverage remains inadequate, evaluate tracklet-level jersey recognition and sport-specific ReID on the same frozen development policy. Use unreadable/unknown, confidence distributions and conflict handling; no score can bypass geometry/component safety. New checkpoints/training require a separate acquisition decision.
- [ ] Evaluate a physical pan/tilt/zoom calibrator or learned NFL landmarks only if the reviewed/propagated geometry baseline fails coverage or annotation cost becomes dominant. Soccer models are research references, not validated NFL drop-ins.

**Final validation:** run the full core test suite, then the evaluation-extra suite; verify real artifacts and references before updating evidence. Preserve failed/unavailable gates explicitly. This milestone proves within-game anonymous replay identity only; schedule different-game testing separately.

## Execution readiness and stop conditions

Tasks 1–3 can start immediately without calibration annotations. Task 4 can generate review material immediately; tasks 5–8 can develop against fixtures while annotations are reviewed. Real acceptance remains blocked until independent geometry, timing and identity labels exist. The user has supplied footage useful for producing them; this plan does not pretend raw video is already reviewed calibration data.

Do not stop all work for missing annotations: continue schema, importer, regression and export work. Stop promotion when a gate fails, an identity component is contradictory, a view lacks valid calibration, alignment is unsupported, or the evaluation denominator is unreviewed. Record the exact next label or defect needed to proceed.
