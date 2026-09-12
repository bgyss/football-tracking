# Goal: integrate and evaluate opt-in McByte tracking

Use this document as the goal prompt for a new task in the football-tracking repository. This is an implementation-and-evaluation contract; completing the adapter alone does not prove improved identities.

## Objective and sources of truth

Integrate Roboflow McByte as an optional local tracker and build a reproducible comparison against ByteTrack and the current BoT-SORT default using the same cached real RF-DETR detections. Determine whether mask assistance reduces within-shot identity switches in All-22 football, and report quality, runtime and memory costs without overstating incomplete evidence. Keep BoT-SORT as default.

Start by reading applicable repository instructions, then:

- `README.md`
- `docs/mcbyte-evaluation-plan.md`
- `docs/research/mcbyte.md`
- `docs/evaluation-plan.md`
- `docs/system-design.md`
- `docs/annotation-and-data-generation.md`
- `docs/superpowers/plans/2026-09-10-football-tracking-implementation.md` for baseline invariants, not as unfinished work to restart

The plan/prompt may be uncommitted in the originating worktree. Before implementation, verify both documents and the research note are present in this task's checkout; transfer those documents from the originating checkout if necessary without overwriting unrelated work. Record the actual starting commit and working-tree changes.

## Baseline and prerequisites

At planning commit `b8f14896d041746223e0cc4df082bc2595b57b3c`, the dependency pins are RF-DETR 1.10.1, trackers 2.6.0 and supervision 0.30.2. Version 2.6.0 already contains McByte: verify its installed API and optional mask dependencies before considering an upgrade. The project has lazy tracker adapters, native-frame iteration, per-shot construction, cached detections and exports, but no standard identity evaluator. Its tracker benchmark uses synthetic detections regardless of the detector probe.

The README's full-clip BoT-SORT/ByteTrack counts are historical integration evidence, not tracking ground truth. The supplied sample is documented as 1,424 frames at 60000/1001 fps with a cut at frame 712. The planning worktree has no video, detection cache, weights or reviewed identity annotations. Locate and validate existing authorized local assets; record missing inputs and continue work that does not depend on them.

## Implementation scope

1. **Dependency and device preflight.** Keep the existing pins if compatible. Add a separate optional McByte/mask dependency extra and update `uv.lock` reproducibly, without making heavyweight model packages mandatory for the core. Record resolved model libraries, checkpoint sources/hashes/licenses, device support and download behavior. Set `enable_mask_manager=True` explicitly for the full McByte variant (upstream defaults to masks off). Make checkpoint/device selection explicit and fail clearly for unavailable requested configurations. Reuse available authorized local weights. If additional assets are required, identify the exact source and size before requesting missing authorization; do not let initialization implicitly fetch models.
2. **Adapter and CLI.** Inspect `src/football_tracking/tracking.py` and `cli.py`. Add `--tracker mcbyte` through `TrackerAdapter`, preferably a dedicated adapter to contain its frame/device/mask contract. Keep package imports lazy. Verify the pinned constructor/update/reset signatures; do not copy the generic adapter's broad `TypeError` retry pattern if it could hide a model error or drop settings. Convert BGR to contiguous RGB exactly once for McByte, preserving BGR for current consumers. Feed every native frame, including explicit empty detections. Preserve original boxes, frame indices, PTS/time base and shot-local tracklet namespaces. Reset all masks, temporal memory and tracker state at each cut; reuse immutable weights only if the backend safely permits it. Keep predicted/mask-propagated states distinguishable from detector observations if they are exported, with backward-compatible schema handling.
3. **Reproducible cache replay.** Add an explicit shared-cache input for controlled comparisons. Validate complete unique source-frame coverage, source hash, checkpoint content hash, class mapping and detector configuration. Preserve proxy provenance across cache hits. Reject malformed/mismatched/incomplete caches instead of silently replacing data or substituting empty detections. Keep detector cache identity independent of tracker settings, but include all tracker/mask settings in run identity/manifests. Do not overwrite another variant's results.
4. **Comparison and evaluator.** Retain the existing synthetic smoke benchmark and add a clearly separate real-cache lane. Implement all supported variants and controls in `docs/mcbyte-evaluation-plan.md`, including effective mask-on versus mask-off reporting. Add a pinned standard tracking evaluator (for example TrackEval) behind an optional dependency and an importer for the reviewed MOT-style/CVAT-derived reference format actually used. Reject unreviewed or absent reference data for quality scoring. Freeze source-frame sampling, visibility/ignore policy, sequence namespaces, confidence filtering, evaluator configuration and development settings. Score raw shot-local tracklets before correction/resolution; do not treat unannotated frames as empty ground truth.
5. **Evidence and plots.** Produce standard per-shot/aggregate HOTA, DetA, AssA, IDF1, ID switches and fragmentation, plus predeclared occlusion-recovery event measurements and coverage. Instrument cold start, tracking/masks, export, total wall time, peak memory, effective device and any fallback. Surface runtime mask disabling; degraded runs cannot pass the mask-active gate. Generate the comparison plots and synchronized failure clips specified in the plan only from actual observations. Export normal pipeline artifacts and a machine-readable report to ignored `artifacts/mcbyte-evaluation/`; write a concise checked-in `docs/evidence/mcbyte-evaluation.md` summarizing provenance, results, commands and decisions.
6. **Documentation and handoff.** Update README usage and relevant evaluation/research notes to match implemented CLI flags and actual behavior. Document a reproducible command for the real comparison and each unavailable prerequisite. Leave the default tracker unchanged and summarize all remaining gates.

## Focused verification

Add meaningful tests for RGB channel handling, exact frame/PTS preservation, empty-frame temporal advancement and expiry at 60000/1001 fps, shot-cut mask/state reset, ID uniqueness, optional-dependency absence, explicit device/weight failures, and effective mask-fallback reporting. Verify cache rejection for wrong video/checkpoint, missing/duplicate frames and cached proxy data. Test evaluator frame-number conversion, ignored/unlabeled intervals and shot namespaces using small analytically known perfect/swap/gap sequences. These establish correctness, not football accuracy.

Run the existing repository commands from the project root:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_tracking.py tests/test_cli.py tests/test_detector.py -q
UV_CACHE_DIR=.uv-cache uv run pytest -q
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking --help
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run --help
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking benchmark --help
git diff --check
```

After defining the optional McByte/evaluation extras, document and run their exact install/import smoke commands. Run added cache/evaluator tests as part of the full suite. Execute a short real mask-active smoke test and then the full sample comparison when verified assets and a supported local device are available. Record commands and outputs, not just intended checks. Inspect output frame count, source-frame/PTS mapping, duration within one source-frame interval, audio sync and both sides of cut 712. Review both complete shots and difficult contact/recovery intervals before making a quality claim.

## Acceptance and completion states

- **Plumbing ready:** the opt-in adapter, configuration, strict shared cache, provenance, evaluator and reports are implemented; focused and full tests pass; existing tracker/export behavior remains usable without mask dependencies.
- **Real integration verified:** the actual pinned backend processes the full source with verified active masks, source/export invariants hold and effective device/resources are recorded. Mocked backends or mask-disabled fallback do not satisfy this state.
- **Quality evaluated:** reviewed references, frozen evaluator settings, per-shot metrics, failure examples and plots support a decision under the proposed rules in the evaluation plan. Report failed targets as failed, not blocked. Report missing reference data as `not_evaluated`.
- **Promotion candidate:** existing per-shot identity gates and the proposed comparative switch-reduction/non-regression rules pass. Do not change the default in this task; the broader game-disjoint held-out gate remains explicit.

Report these states independently. The task's integration-and-development-evaluation objective is complete only after real integration and quality evaluation have evidence, even if the measured conclusion is that McByte should not be adopted. Missing hardware/assets/human-reviewed labels permit a plumbing handoff with precise blocked gates, not a claim of complete evaluation. A negative quality finding is a valid experimental result.

## Scope and execution boundaries

Work locally on the assigned checkout, preserve unrelated changes and keep documentation paths repository-relative. This goal authorizes code changes, ordinary dependency installation, local tests and local evaluation using available authorized assets. Do not train or fine-tune RF-DETR, add cross-view replay matching, change calibration, add Astra/hosted inference, upload footage, start paid/cloud compute or acquire additional footage as part of this task. Keep raw footage, checkpoints and bulky generated outputs out of Git. Do not commit, push, merge or publish unless separately requested.

If a prerequisite prevents progress, record its exact owner/input/error and the command needed to resume; finish independent authorized work first. Do not replace the requested backend with IoU or synthetic detections and call it a McByte success. Do not lower predeclared gates after examining evaluation results.
