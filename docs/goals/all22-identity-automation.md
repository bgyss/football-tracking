# Implementation handoff: minimally supervised All-22 cross-shot identity

Prepared 2026-09-24 from the
[research document](../research/all22-cross-shot-identity-2026-09-24.md).
Status: **proposed work; no implementation or new accuracy result**.

## Copy-ready task prompt

> Implement the bounded All-22 identity-automation program in this document,
> starting with A0 and advancing through applicable slices in dependency order.
> Read `AGENTS.md`, `CLAUDE.md`, the linked research, `docs/evaluation-plan.md`,
> `docs/human-review-all22.md`, and the latest committed evidence first. Audit the
> actual checkout and available data; do not assume ignored assets from another
> worktree exist here. Preserve unrelated work.
>
> The objective is automatic same-play identity across sideline/end-zone replay
> views with minimal recurring human effort. Produce automatic geometry, timing
> and identity predictions, an uncertainty-ranked exception queue, and measured
> accuracy/coverage/labor evidence. Keep anonymous shared-player identity separate
> from named roster identity. Reuse the existing pipeline and deterministic
> artifact contracts. Do not restart with a standalone demo tracker.
>
> Keep human-reviewed truth separate from machine predictions. Introduce an
> opt-in, versioned inference policy instead of setting `reviewed: true` on model
> output or disabling current validation. Automatic proposal generation may
> proceed while evaluation labels are unavailable; real-accuracy claims and
> promotion must wait for independent evidence. Another model's agreement is
> not an independent human-review record.
>
> Deliver small validated slices, starting with a diagnostic baseline and
> automated NFL geometry/timing proposals. Prefer local execution; design one
> portable GPU-worker adapter after the local job contract works. Prepare a sparse
> Astra challenger and an optional geometry-assisted reranker, disabled by default.
> Only add heavy models justified by the measured bottleneck. Keep BoT-SORT as the
> default and heavy imports optional.
>
> Follow existing authorization boundaries. A request to implement authorizes
> local coding and ordinary validation, not footage upload, paid API/GPU usage,
> checkpoint/data acquisition, remote CI, publishing, pushing, PR creation or
> training at scale unless explicitly included in that execution request. Prepare
> exact manifests, source ranges and resource/spend caps before requesting any
> missing remote-run authorization. Continue independent local work when an
> external gate is unavailable; do not manufacture its evidence.
>
> Report what changed, tests actually run, realized experiments, source/provenance
> hashes, remaining unknowns and the next bounded slice. Do not declare completion
> from synthetic fixtures, a rendered overlay, low fit residuals, zero returned
> matches or an unscored model assertion.

The prompt above is for a subsequent implementation task. This research task does
not execute it. The following slices form one program, not seven parallel
rewrites. Complete the first real-data checkpoint before expanding the model stack.

## Baseline and invariant contracts

At inspected revision `502115a`, the September 23 report records actual local
RF-DETR/BoT-SORT observations but no attempted cross-shot resolution, calibration,
reviewed play-time map or evaluated identity accuracy. This checkout has no
ignored `data/` or `artifacts/` assets. Verify their current availability before
running inference. Do not confuse the 1280×720 sample with the separately
documented 1920×1080 full game.

Preserve:

- Original source hash, zero-based frame number, integer PTS and time base;
  crop/review/worker indices must round-trip through an explicit source map.
- Separate shot, play, tracklet, anonymous player and optional roster identifiers.
- Fresh image-space tracking per shot and no replay double-counting.
- Immutable raw detector/tracker rows, with explicit derived repair overlays.
- Deterministic sorted outputs, strict cache provenance, missingness and abstention.
- Source/checkpoint/config hashes and execution mode, including proxy/fallback.
- Explicit model acquisition, bounded memory and optional heavy dependencies.
- Independently reviewed reference truth, separate from pseudo-labels and outputs.

Useful current seams:

| Concern | Existing files |
| --- | --- |
| Ingest/source mapping | `video.py`, `cache.py`, `cvat.py`, `annotations.py` |
| Geometry | `field.py`, `calibration.py`, `calibration_timeline.py` |
| Time/candidates | `replay.py`, `scripts/propose_timing_events.py` |
| Local tracks/cues | `tracking.py`, `tracklet_refinement.py`, `identity_cues.py` |
| Cross-view assignment | `identity.py`, `identity_resolution.py` |
| Orchestration/export | `cli.py`, `schema.py`, `export.py`, `validation.py` |
| Review/evaluation | `identity_review.py`, `evaluation.py`, `scripts/check_identity_readiness.py` |

Module filenames in the table are relative to `src/football_tracking/` unless a
`scripts/` prefix is present. New modules below are proposed interfaces; select
final names after checking the current checkout.

## A0 — Reproducible diagnostic and reference slice

**Objective:** establish which missing stage limits the current clip and prepare
the smallest source-addressed real-data comparison.

1. Record revision, working-tree status, source/weight/cache availability and
   hashes. Reuse valid existing caches. Record hardware and effective device.
2. Produce per-view counts at raw detection, filtering, tracker association and
   export. Explain dropped/unmatched detections, sideline/referee candidates and
   tracklet splits. Counts alone are not precision/recall.
3. Select one shared-action interval with context before/after it. Prepare full
   views, track galleries, field landmarks and proposed timing events. Source
   frame/PTS addressing is mandatory.
4. Prepare independent reference tasks: fit and spatially/temporally withheld
   landmarks in both views; separated matching events plus held-out events;
   shared-player IDs, hard negatives, unknowns and within-shot purity labels.
   Use the existing CVAT bridge, including independent second-review records.
5. Run the oracle bottleneck ladder once reference inputs exist: gold
   geometry/timing/tracks, then substitute automatic components one at a time.

**Outputs:** ignored `artifacts/all22-automation/<run-id>/diagnostics.json`,
review bundles, source manifest, fixed evaluation scope, measured review-time
baseline and experiment ledger. Commit a compact evidence report only after a
real run; use neutral paths in docs.

**Acceptance:** all examined rows/crops resolve to source frames; counts reconcile;
reference and proposals remain distinguishable; each unavailable resource is
explicit. A0 preparation can finish without labels, but A0 measured accuracy cannot.

**Do not:** rerun every detector merely because a worktree lacks artifacts, infer
22 visible players per frame, or call the existing nine-box median measured recall.

## A1 — Typed automatic predictions and an inference policy

**Prerequisite:** A0 source/provenance contract. Independent labels are required
to calibrate and promote acceptance, not to implement the prediction lane.

Introduce schemas distinguishing `proposal`, `machine_validated`, `human_reviewed`,
`rejected` and `unknown`, using additive/versioned artifacts rather than breaking
existing frozen observation contracts. Keep source, producer and quality state
separate. The proposed names need not replace existing public status strings.

An automatic artifact must record:

| Group | Required content |
| --- | --- |
| Source | Source hash, shot/play scope, frame/PTS map, crop transforms |
| Producer | Code revision, model and checkpoint hashes, parameters, run ID |
| Geometry | Hypothesis IDs, orientation/yard-offset alternatives, support interval/polygon, residuals and uncertainty |
| Timing | Source-to-play map, valid support, offset/rate hypotheses, uncertainty and discontinuities |
| Validation | Policy version, internal checks, threshold origins, independent validation report reference if available |
| Decision | Accepted/abstained/rejected with reasons, alternatives and evidence IDs |

Add an opt-in entry point for automatic inference and keep the current reviewed
alignment/reference loaders unchanged. An unvalidated policy may generate
experimental predictions for evaluation but must not label them production-ready.
Only a measured policy can automatically accept supported-domain results.

**Acceptance:** automatic metadata cannot forge review status; existing reviewed
inputs keep their meaning; missing/stale/mismatched artifacts fail closed;
experimental outputs report `not_evaluated` without truth. Test cross-source
substitution, unsupported intervals and backward compatibility.

## A2 — Automatic NFL field registration

**Prerequisite:** A1 proposal schema and A0 source frames. Use existing field
geometry; no new model is required for the first line-topology baseline.

Implement a pluggable field-feature proposer. First use line/hash/number-context
features and robust point/line fits. Enumerate yard-offset/reflection hypotheses
and preserve unresolved semantics. Fit each view separately. Add motion-triggered
keyframes, field-only propagation and absolute relocalization. Distinguish fit
residuals, automatic consistency checks and independent withheld accuracy.

Then compare PnLCalib-style learned features/refinement if baseline failures
justify adaptation. Importing soccer field constants is a failing implementation.
BroadTrack-style physical camera constraints are a later bounded ablation, not a
requirement to port an entire upstream stack.

**Outputs:** field feature proposals, hypothesis-ranked calibration timeline,
support/uncertainty maps, diagnostics overlays, failed-hypothesis reasons, per-view
withheld residual report when reference exists.

**Benchmark-promotion acceptance:** on independently labeled development/held-out
keyframes, no propagation across cuts; no absolute yard claim from an ambiguous
repeated grid; median ≤1 yard and p95 ≤2 yards for an eligible measured fit;
missing feet/contact uncertainty stays separate. Include
wrong-five-yard, mirror, degenerate points and drift tests. If gates fail, emit
proposals and a targeted correction item rather than relaxing them.

## A3 — Automatic play grouping and time mapping

**Prerequisite:** A1, with A2 field trajectories when available. Coarse event
proposals and sparse VLM packet preparation can proceed before A2 is complete.

Implement coarse motion/event windows and same-play candidate ranking. Fit bounded
offset/rate maps from team-level field occupancy, motion and event sequences before
assuming player IDs. One snap implies only an offset if playback rate is known;
fit two separated correspondences for rate and reserve an additional event to
validate. Refine locally at native source cadence.

Detect freeze/duplicate frames and edits. Add piecewise maps only when needed,
with monotonicity/rate bounds and invalid discontinuities. Never extrapolate into
unseen overlap or use unrestricted warping to make wrong plays agree.

**Outputs:** source-addressed event proposals, ranked time maps, supported overlap,
rate/offset uncertainty, withheld residuals and unresolved-play reasons.

**Benchmark-promotion acceptance:** correct offset/rate on controlled fixtures;
wrong-play/no-overlap and freezes abstain; independently labeled held-out events
meet the existing 50 ms default policy before the timing method is accepted for
automatic use. No gold event participates in both fitting and validation. Report
view-specific native-frame and millisecond errors. After this policy is validated,
unseen production plays use its frozen support, uncertainty and internal-consistency
checks; they do not require newly labeled human events per play. Unsupported or
low-confidence maps still abstain and enter the exception queue.

**Optional escalation:** reproduce VisualSync on a small constant-speed pair only
if local timing remains limiting and a GPU job is authorized. Its offset solver
does not satisfy variable-speed replay support by itself.

## A4 — Track purity, joint identity and sports cues

**Prerequisite:** source-compatible tracks plus A2/A3 predictions or oracle inputs
clearly labeled as such. A0 independent truth is needed to select a winner.

1. Diagnose and split mixed tracklets through immutable overlays. Test an offline
   split/link baseline such as GTA-link only if fragmentation/purity warrants it.
2. Reuse the existing constrained assignment; add quality-aware field/time costs,
   competing global-assignment margins, unmatched outcomes and component checks.
3. Add a compact sports OSNet crop-gallery adapter as the first appearance
   comparator. Hard negatives must include same-team neighbors.
4. Add legibility-aware JNR with per-crop number distributions and unknown states.
   Compare public uncertainty-JNR weights or the PARSeq sports pipeline only after
   checking licensing and checkpoint access. Preserve temporal correlation.
5. Make automatic cue influence explicit under a new experimental policy.
   Current `review_rank_score` cannot be relabeled as an accepted-match score.
6. Consider joint timing/geometry/assignment refinement only after the modular
   baseline; cap iterations and preserve alternative solutions and independent
   validation. Learned probabilities require calibration on unused development data.

**Outputs:** candidate feature table, rejected edges, assignments, final
components, cue galleries, uncertainty/coverage curves and ablation report.

**Acceptance:** no simultaneous same-view duplicate identity; no chain-induced
component contamination; incompatible teams/reliable reviewed numbers reject;
multiple nonoverlapping fragments can retain one identity with evidence; named
identity remains optional. Compare geometry only, appearance only, geometry plus
jersey, geometry plus ReID and combined. Require zero observed false merges and
≥0.80 shared-player coverage on the sample before claiming that gate passed.

## A5 — Sparse Astra/VLM experiment

**Prerequisite:** A1 and source-addressed shot-local tracks. The independent
challenger does not need geometric candidates; assisted reranking needs A2/A3.

Implement a provider-neutral packet builder and offline response importer first.
Use the research document's image/response contract. Store target and multiple
candidates, full-view context, nonredundant crops, timing metadata and an unmatched
option. Add a source-limited request for better crops. Persist exact prompts,
images or image hashes, responses, provider/model IDs and usage.

Prepare a strict structured-output adapter for the currently documented Astra
API, disabled by default. Verify docs/API model availability at implementation
time. Do not assume this task's model access implies API credentials or GPU access.
Test raw image-only identity proposals separately from geometry-assisted choices.
One open-weight or hosted alternative can be selected as a comparator.

**Outputs:** immutable packets, dry-run cost assumptions, validated proposal
responses, real usage ledger if authorized, blinded comparison and failure cases.

**Acceptance:** unknown IDs/frames, stale source, malformed output, hallucinated
coordinates, refusal, timeout and exhausted budget all fail safely; local inference
still completes. Promotion requires improved coverage or labor at the same
false-merge constraint. Model outputs never alter the reference labels.

**External gate:** absent API authorization/credentials, finish packet generation,
mock/failure-path tests and importer work; report the real inference comparison
as not run. Do not replace a missing API result with a fabricated example score.

## A6 — Portable server worker and optional Astra job control

**Prerequisite:** reproducible local job contract. Select one backend, not several.

Package the same source/cache/model contract into a pinned worker environment.
Choose an existing Linux GPU server, Hugging Face Jobs, Modal or Runpod according
to access and the chosen models. Export source-addressed observations and raw
cues, not only an annotated movie. Persist outputs before worker exit.

Proposed job record includes source URI/hash/ranges, code/container digest,
checkpoint hashes, detector cache identity, inference policy, stages requested,
resource limits, timeout, maximum retries, spend cap, result URI and idempotency
key. Keep credentials out of the artifact and prompt. A source file must actually
be staged to the authorized worker; reject inaccessible or hash-mismatched inputs.

Provide CLI submit/status/cancel/results first. Optional MCP/function tools may
then expose those bounded operations to Astra or the implementation model. This
does not require an LLM to execute every job. Log the distinction between worker
execution success and identity/evaluation success.

**Acceptance:** duplicate submit does not create a duplicate job; interrupted work
resumes compatible stages; cancellation works; outputs survive worker shutdown;
missing weights fail explicitly; source/PTS/crop transforms match local results.
Compare numerical/model nondeterminism with declared tolerances and report actual
hardware, wall time, peak RAM/VRAM, cache hits, billed usage and cost.

**External gate:** prepare the container/configuration and bounded job manifest
before requesting any missing upload/spend authorization. Do not start a full-game
GPU experiment to find out whether a two-shot job works.

## A7 — Exception-only review and promotion evidence

**Prerequisite:** at least one fully automatic end-to-end configuration and an
independent reference. Maintain separate automatic, VLM-assisted and corrected
outputs throughout evaluation.

Route high-impact geometry/time corrections before individual identity ties.
Present synchronized views, source navigation, alternatives and unknown/ambiguous
choices. Capture actual correction time and provenance. Add random audits of
accepted cases so review-selection bias does not hide confident errors.

Run the frozen experiment matrix from the research. Keep all views of a play in
one split and acquire at least ten additional plays from at least three games for
the existing breadth gate. If only one game is available, state that limitation
and retain the cross-game gate as open.

**Acceptance:** existing accuracy/coverage/reliability targets are measured, not
merely implemented as booleans; denominators and per-view failure strata are
reported; independent labels never enter training or threshold tuning for their
test split. Report exact false merges and component contamination, not only a
single aggregate score.

Proposed labor goals: at least 50% median review-time reduction against A0 and an
eventual 80% of supported-domain plays needing no user intervention. These goals
must accompany identity coverage and quality; abstaining on everything cannot
satisfy the automation objective. Keep them separate from existing accuracy gates.

**Definition of done:** an authorized, reproducible automatic pipeline has measured
results meeting the agreed gates on the required real data, documented unsupported
conditions, and a functioning exception path. If data, review, compute or accuracy
is missing, deliver the completed implementation slice with the corresponding
evidence gate explicitly open. Do not call the overall program achieved.

## Validation commands and evidence discipline

Existing commands below were checked against repository guidance and source
during research; they were not executed here. Run from the project root with the
declared development dependencies installed:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev
PYTHONPATH=src:. UV_CACHE_DIR=.uv-cache uv run pytest -q
git diff --check
```

Use focused tests during each slice, then the full suite after integration. Keep
heavy adapters lazy so the core suite remains independent of model downloads.
Meaningful tests cover transformations/provenance, wrong source/camera, semantic
yard ambiguity, drift, time-rate changes, wrong-play assignment, fragments and
transitive contamination, plus provider/worker failures. Synthetic tests verify
invariants only; they do not satisfy the real-footage exit gate.

With the sample and existing checkpoint available, the current stage command is:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/stage_local_review.py \
  --source data/all-22-lions-rams-sample.mp4 \
  --run-dir artifacts/all22-run \
  --output artifacts/all22-review-stage \
  --manual-cut 712 \
  --detector-checkpoint /path/to/local-rf-detr-small.pth
```

The checkpoint argument is a required local acquisition decision, not an implicit
download. This command produces proposals; it does not solve identity by itself.
For existing reviewed artifacts, preserve the current evaluation-readiness check:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/check_identity_readiness.py \
  --source data/all-22-lions-rams-sample.mp4 \
  --annotations artifacts/all22-reference/reviewed-manifest.json \
  --calibration artifacts/all22-reference/calibration-timeline.json \
  --play-alignment artifacts/all22-reference/play-alignment.json \
  --reviewed-reference artifacts/all22-reference/reviewed-reference.json \
  --output artifacts/all22-reference/readiness.json
```

These artifact paths are proposed destinations and must contain genuine reviewed,
source-compatible inputs before the command is meaningful. New automatic CLI/API
flags do not exist yet: implement and document them rather than pretending these
reviewed-input commands accept machine proposals.

## First handoff checkpoint

The next model should return **A0 diagnostics plus A1 schemas and an A2/A3 proposal
baseline**, with source-addressed examples and tests. Also prepare A5 image packets
if useful to compare Astra early. It should not spend its first slice swapping
trackers or building a multi-provider server platform.

If the source is unavailable, implement/test the source-independent contracts and
report the exact missing source/cache/weight inputs. If labels are unavailable,
generate proposals and review material while leaving accuracy unscored. If geometry
or timing remains ambiguous, preserve hypotheses and show the smallest useful
correction. This keeps implementation moving without hiding why identity is still
unproven.
