# McByte evaluation and integration plan

Status: planned, 2026-09-12. No McByte run or accuracy measurement was performed for this plan. Integrate McByte as an opt-in candidate; keep BoT-SORT as the default until comparative evidence supports a separate promotion decision.

Read the [source investigation](research/mcbyte.md), [project evaluation contract](evaluation-plan.md), and [copy-ready integration goal](goals/mcbyte-integration.md) together.

## Why evaluate it here

The useful hypothesis behind Roboflow's recommendation is that mask propagation can disambiguate overlapping players better than box association alone. Test that hypothesis on same-team crossings, line-of-scrimmage contact, partial occlusion, and recovery from piles in All-22 footage. Better tracking on these cases could improve anonymous trajectories and team aggregation. It does not establish roster identity, cross-view replay matching, or yard calibration.

The pinned `trackers==2.6.0` already includes `McByteTracker`; first try that version with its optional mask dependencies. McByte combines ByteTrack-style association with SAM initialization and Cutie mask propagation. Masks default off: the full variant must explicitly set `enable_mask_manager=True`. Its pinned source expects RGB frames, although the development-page example passes an unconverted OpenCV frame. Use the versioned source as the adapter contract and prove the boundary with a color sentinel test. The code has an out-of-memory fallback that can disable masks; record effective behavior, not just the requested algorithm name. See [verified APIs, dependencies, and limitations](research/mcbyte.md).

## Verified project baseline

Repository inspection at `b8f14896d041746223e0cc4df082bc2595b57b3c` found:

| Area | Current behavior | Work required |
| --- | --- | --- |
| Tracking | `TrackerAdapter` accepts detections, BGR frame, timestamp, frame index and PTS; `RoboflowTracker` supports ByteTrack and BoT-SORT | Add a lazy McByte adapter with its own RGB boundary and explicit configuration |
| Pipeline | CLI builds a fresh tracker at each shot; default is BoT-SORT | Add opt-in `mcbyte`, preserve shot-local IDs and reset all temporal state |
| Dependencies | RF-DETR 1.10.1, trackers 2.6.0, supervision 0.30.2 | Resolve a separate optional mask extra and lock it; verify real installed signatures |
| Cache | Output-local `detections.jsonl`, keyed by video hash and detector configuration | Add explicit shared-cache input and strict completeness/provenance checks |
| Benchmark | `_tracker_benchmark` always constructs `SyntheticDetector`, even when the detector probe is real | Add a real-cache comparison lane; retain the existing proxy smoke lane with clear labels |
| Metrics | `summarize_tracks` reports counts and lengths; no ground-truth evaluator | Add a pinned standard evaluator and a reviewed-reference importer |
| Identity | `stable_anonymous_ids(tracklet_ids, [])` makes no cross-view joins | Compare raw shot-local tracklets before team/identity corrections |
| Assets | No `data/`, `artifacts/`, weights, or reviewed tracking labels in this worktree | Locate existing authorized local assets before any real run; record missing inputs |

The README records a prior 1,424-frame RF-DETR Small run: BoT-SORT yielded 59 tracklets and ByteTrack 50 on the same detections. These are historical integration observations, not current rerun results or identity accuracy. Fewer tracklets can mean either successful recovery or incorrect merges. The tracked metadata specifies 1280 × 720, 60000/1001 fps, 23.757067 seconds; the reviewed cut is source frame 712.

## Staged experiment

| Stage | Action | Exit evidence |
| --- | --- | --- |
| 1. Compatibility | Resolve pinned McByte mask dependencies in an isolated environment; inspect model loading, device support, asset provenance and fallback paths | Exact package/weight versions and hashes, license notes, chosen device, import/initialization result; no implied CUDA or MPS support |
| 2. Adapter | Add opt-in CLI configuration, RGB conversion, frame-by-frame updates, shot reset, explicit errors and effective-mode reporting | Deterministic adapter tests plus a real-backend short smoke run when weights/hardware are available |
| 3. Fair replay | Freeze one real detection cache and replay identical native-rate frames through all candidates | Input/cache/config hashes, full frame coverage, separate outputs, equal detector candidates, successful frame-712 reset |
| 4. Identity scoring | Score reviewed references under the existing cadence/ignore/split rules | Per-shot and aggregate HOTA, DetA, AssA, IDF1, ID switches, fragmentation, recovery and coverage; failure clips |
| 5. Decision | Compare measured identity benefit against compute/memory cost and failure behavior | Written retain/reject/promote-candidate decision; held-out generalization remains a separate gate |

Missing model assets or annotations do not stop independent adapter/evaluator work. They do prevent completion of the corresponding real-run or quality gate.

## Controlled comparison matrix

All primary rows use the same real RF-DETR detections, player filtering, original frames, shot boundaries, hardware, reference and evaluator configuration. Run tracking at native cadence; sample predictions for scoring afterward. Feeding only the approximately 10 Hz scoring frames into a temporal mask model would change the experiment.

| Variant | Purpose |
| --- | --- |
| ByteTrack, existing pinned baseline | Establish the simplest box-association comparator |
| BoT-SORT, camera compensation enabled | Compare against the current product default |
| BoT-SORT, same configuration with compensation disabled | Isolate camera compensation within BoT-SORT |
| McByte, mask assistance enabled and verified effective | Test the proposed identity benefit |
| McByte, masks disabled through a supported setting | Ablate mask assistance within McByte; distinguish from plain ByteTrack |

Match directly comparable thresholds and lost-track durations where semantics agree, and record differences. Start with documented defaults; if tuning is needed, use a bounded, recorded search on development plays with comparable effort per tracker. Freeze all settings before held-out scoring. A mask-disabled McByte run is not a replacement for the separately measured ByteTrack baseline. If the ablation is unsupported, mark it unavailable rather than patching the algorithm invisibly.

Keep team resolution and manual corrections out of the primary comparison. A secondary identical team-resolution pass may measure assigned-team accuracy and coverage against reviewed labels. Do not introduce detector fine-tuning, Astra calls, appearance ReID, calibration changes or cross-view joining into this experiment.

## Reference and replay integrity

Use the [local annotation workflow](annotation-and-data-generation.md). Human-review persistent anonymous IDs, player boxes, visibility, ignore regions and ambiguous intervals; model-generated preannotations are not accepted ground truth. Densely annotate both complete shots at the declared approximately 10 Hz evaluation timestamps, then inspect native-rate transitions around contact. Preserve the frozen evaluation frame list and a reversible mapping between source indices/PTS and any evaluator-specific one-based frame numbers. Evaluate each shot as a separate sequence so IDs cannot leak through the replay cut.

Build native-rate labeled event windows for brief swaps, occlusion recovery and same-team crossings. Define the event denominator, maximum recovery gap, incorrect-recovery rule and ambiguity policy before scoring. Report counts and denominators separately from the standard sampled-cadence metrics. Do not average windows as independent games or count two views of one play as independent samples.

Strengthen cache validation for this experiment: require one explicit record per decoded source frame, including empty detections; reject duplicate, missing or out-of-range frame records. Check source hash, actual checkpoint content hash, class mapping, detector threshold/preprocessing and proxy/real provenance. The current configuration hashes the checkpoint path rather than its contents, and cached synthetic runs can lose their proxy flag because no detector instance is created. Fix these provenance gaps before making comparison claims. Never silently regenerate detections or treat a missing cache record as an observed empty frame in the controlled lane.

## Measurements and proposed decision rules

The existing project targets remain unchanged: development IDF1 at least 0.90 and at most one ID switch per shot; report HOTA, DetA, AssA and fragmentation. Detector precision/recall at IoU 0.5 remain separate upstream gates. Mask-assisted association cannot excuse missing-player recall.

Use the following conservative, **proposed** comparison rules, fixed before seeing results:

- Integration passes only when all adapter/cache/export invariants pass and a real mask-active run is evidenced. Tests with mocked backends establish only plumbing readiness.
- A development quality candidate must meet the existing per-shot identity targets, show strictly fewer total ID switches than both ByteTrack and BoT-SORT, and have no per-shot regression in IDF1 or HOTA against either primary baseline. Report raw values and differences, not only a pass flag.
- If a baseline already has zero switches, that comparison cannot demonstrate switch reduction. Report no demonstrated reduction; do not weaken the rule after observing scores. Wider data may resolve a tie.
- Missing reviewed labels means quality is `not_evaluated`; lower track counts, good overlays and synthetic metrics cannot substitute. A failed run or silent mask fallback cannot count as a successful mask-assisted result.
- Leave BoT-SORT as the default in this integration task. A broader adoption decision requires the existing held-out breadth target: at least ten additional plays from at least three games, game-disjoint development/evaluation, replay-grouped splits and frozen settings. Report per-game/per-play results and uncertainty at the play or game level when the sample supports it.

These are planning criteria, not measured claims or a statistically established minimum effect size. Record total wall time, wall seconds per video second, detector time, model initialization, mask processing/association, export time, peak host/device memory, device fallback and exceptions. Avoid double-counting nested stage timers. Separate cold start from steady-state measurements; record warm-up/repetitions and synchronize accelerator timing. Report runtime cost alongside quality; no runtime ceiling or real-time claim is implied by the offline project.

## Review artifacts

Store generated media and bulky/raw results under ignored `artifacts/mcbyte-evaluation/`. Produce one machine-readable comparison report with explicit `integration`, `real_backend`, `quality` and `held_out` statuses, run manifests, standard evaluator output/configuration, reference hashes and failure examples. Preserve each variant's existing video/table outputs.

Required plots, once real labels/results exist:

1. Per-shot ID switches and IDF1/HOTA for each variant, with raw counts and a clear common metric scale.
2. Identity quality versus wall seconds per video second, annotated with device and whether masks stayed active.
3. Native-frame timelines for adjudicated crossings/piles showing ground-truth identity, predicted ID, gaps and switches, with synchronized side-by-side clips.

No numerical plot is produced at planning time because there are no McByte measurements to plot. Check in a compact summary at `docs/evidence/mcbyte-evaluation.md` with artifact paths/hashes, commands, decision and remaining gates. Keep private footage and model binaries out of Git.
