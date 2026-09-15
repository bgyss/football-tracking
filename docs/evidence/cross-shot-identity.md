# Cross-shot identity: measured state and blocked gates

Branch: `feat/cross-shot-identity`. Implementation planned in `docs/superpowers/plans/2026-09-14-cross-shot-identity.md`.

Command used to regenerate the evidence cited below:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run \
  --input data/all-22-lions-rams-sample.mp4 \
  --output artifacts/cross-shot-evidence \
  --detector synthetic --tracker iou --manual-cut 712
```

`artifacts/` is gitignored; the numbers below were read from artifacts on disk after this
run and cross-checked against the pre-change baseline and other committed runs. They are
evidence this document cites, not files committed alongside it.

## The question that prompted this work

The user reported that player identities "weren't locking in" across the cut from the
sideline view to the end-zone replay at frame 712, in a clip processed with the McByte
tracker. The finding below is that this had two separate causes, and McByte is not one of
them.

## Finding 1: McByte was not the cause

`artifacts/mcbyte-evaluation/mcbyte-mask-on-unbounded/metrics.json` records, for both
shots:

```json
{
  "shot_id": "shot-0", "masks_active": true, "fallback_reason": null, "consecutive_mask_failures": 0
}
{
  "shot_id": "shot-1", "masks_active": true, "fallback_reason": null, "consecutive_mask_failures": 0
}
```

That run produced 18,343 observations over 52 tracklets
(`track_summary.observation_count` / `track_summary.unique_tracklets`), with
`track_summary.duplicate_track_ids_within_frame: 0`. McByte's masks were active
throughout, with no fallback and no consecutive mask failures. The tracker was doing what
it was supposed to do.

## Finding 2: cross-shot identity was never implemented, for any tracker

Before this branch, `cli.py` passed a hardcoded empty link list to
`stable_anonymous_ids`, so `identities.json` was always a 1:1 map from shot-local
tracklet ID to player ID — no player ID ever spanned more than one tracklet, regardless
of tracker. This is why identities did not "lock in" across the cut: there was no code
path that could join them, independent of tracker choice or mask quality.

Measured across three committed runs (read from each run's `identities.json`, keyed by
counting IDs whose prefix is `shot-0:` versus `shot-1:`):

| Run | Tracklets | Unique player IDs | shot-0 / shot-1 | IDs spanning >1 tracklet |
| --- | --- | --- | --- | --- |
| `artifacts/mcbyte-evaluation/mcbyte-mask-on-unbounded/identities.json` | 52 | 52 | 24 / 28 | 0 |
| `artifacts/mcbyte-evaluation/mcbyte-mask-off/identities.json` | 52 | 52 | 24 / 28 | 0 |
| `artifacts/verified-botsort/identities.json` | 59 | 59 | 27 / 32 | 0 |

In every run, unique player IDs equal tracklet count and no ID spans more than one
tracklet: cross-shot identity resolution never ran.

## The default path is unchanged by this branch

Rerunning the real clip after this branch's changes, without `--play-alignment`:

- `artifacts/cross-shot-evidence/identities.json` is byte-identical to
  `artifacts/cross-shot-baseline/identities.json` (the pre-change baseline) and to
  `artifacts/cross-shot-verify2/identities.json` (a controller-verified post-change run).
- `artifacts/cross-shot-evidence/identity-links.json`:

  ```json
  {
    "status": "not_attempted",
    "reason": "--play-alignment was not provided",
    "links": [],
    "shot_count": 2,
    "tracklet_count": 4,
    "player_id_count": 4,
    "cross_shot_player_ids": 0
  }
  ```

The tracklet count is 4 (2 per shot) because this run uses the synthetic proxy detector,
which emits fixed boxes rather than real detections. This is a plumbing run confirming
the default path is untouched, not a detection result — do not read `tracklet_count: 4`
as a claim about how many players are visible in the footage.

`artifacts/cross-shot-evidence/tracking-evaluation.json`:

```json
{
  "status": "not_evaluated",
  "reason": "--reviewed-reference was not provided"
}
```

## What this branch built

A cross-shot identity evaluator reporting precision and coverage separately
(`evaluation.py`), an `identity-links.json` evidence file written by every run
(regardless of whether cross-shot resolution was attempted), a reviewed play-time
alignment loader and a cross-shot candidate scorer (`replay.py`), and CLI wiring behind
`--play-alignment` that abstains without calibrated field positions. See
`.superpowers/sdd/progress.md` for the task-by-task record and the decisions made while
building it (including the injective-pairing fix and the documented limitation that only
the first two aligned shots are resolved, with the rest recorded in
`identity-links.json`'s `unresolved_shots`).

None of this has been exercised end-to-end on the real clip: doing so requires a reviewed
play-time alignment and shot-specific calibration landmarks, neither of which exists for
this footage yet (see "Blocked gates" below). `--play-alignment` and `--calibration` were
exercised only against synthetic fixtures in the test suite, not against
`data/all-22-lions-rams-sample.mp4`.

## The critical caveat: two failure modes remain indistinguishable

An ID that fails to persist across the cut could be caused by either:

1. A missing cross-shot join (the case this branch addresses), or
2. Within-shot fragmentation — the same physical player receiving more than one
   tracklet ID inside a single shot, before any cross-shot question arises.

`tracking-evaluation.json` reports `cross_shot.status: not_evaluated` because no reviewed
reference with a `cross_shot_identity` map has ever been supplied for this clip. Without
that reference, the artifacts in this repository cannot separate these two failure modes.
This is the most important limitation of the current evidence: a future run that still
fails to "lock in" identities across the cut could be failing for either reason, and
nothing measured so far tells us which.

## Blocked gates

The following prerequisites, copied verbatim from the milestone section of
`docs/superpowers/plans/2026-09-14-cross-shot-identity.md`, are not yet satisfied for
the real clip:

| Prerequisite | Blocks | Why |
| --- | --- | --- |
| Reviewed MOT-style reference with global identity map | Measured baseline, Resolution verified | `evaluate_tracking` refuses unreviewed data by design (`evaluation.py:45`) |
| Reviewed snap-frame anchors for both shots | Resolution verified | Play-time alignment has no other anchor |
| Shot-specific calibration landmarks for both shots | Resolution verified | Field-position agreement is the only view-invariant geometric signal; image coordinates are meaningless across a cut |

Until calibration landmarks exist for both shots, the resolver is expected and required
to abstain on the sample clip. A measured abstention is a correct outcome for this
milestone's plumbing state, not a failure or a success — it is what the code is supposed
to do without its inputs.

## What this document does not claim

No HOTA, IDF1, coverage, or precision number appears anywhere in this document, because
no artifact read for this document contains one. The predeclared cross-view gate in
`docs/evaluation-plan.md` is:

> Zero false merges in the sample; at least 0.80 coverage of human-resolvable shared
> players

That gate has not been evaluated, let alone met, on this footage — `cross_shot.status` is
`not_evaluated`, not `passed` or `failed`. This document does not show the cross-shot
resolver working correctly on real footage; it shows that the plumbing exists, that it
correctly abstains without the inputs it requires, and that it leaves the previously
unimplemented default path unchanged.

## Conclusion

Cross-shot identity was never implemented, for any tracker, before this branch. McByte's
mask pipeline was healthy throughout and is not implicated in the reported symptom. The
resolver built on this branch abstains, by design, until reviewed play-time anchors and
shot-specific calibration landmarks exist for both shots of this clip.

## 2026-09-15 correctness audit

This update inspects commit `156f751` and supersedes any inference that supplying
landmarks alone is sufficient for correct identity. Earlier artifact measurements above
remain historical; they were not rerun in this worktree. The implementation now exists,
but its real-footage acceptance gate remains unproven.

The [implementation plan](../superpowers/plans/2026-09-15-calibration-identity-correctness.md)
and [primary-source research](cross-shot-identity-research-2026-09-15.md) describe the
next work. No pipeline implementation was changed during this audit.

### Confirmed with executable probes

| Probe | Actual result | Interpretation |
| --- | --- | --- |
| Two left/two right tracklets; every score is 0.9 | Both assignments return `same` | Selected columns are excluded from ambiguity alternatives; a completely tied assignment is accepted. |
| Nine image/field landmarks, field scale 0.1 yards/pixel, center displaced 1.5 yards | `median_error_px` = 0.20208099677691288, identical to computed yard residual; all nine inliers, `is_valid=True` | The field-space residual is mislabeled as pixels. |
| Two human-resolvable players shared by two views; only one player detected and correctly linked | `coverage=1.0`, `resolvable_pairs=1`, both gates true | Prediction-derived pair counting hides the missing shared player. Correct player coverage for this fixture is 1/2. |
| Perfect detection boxes with two IDs alternating over three frames | Four switches, aggregate `IDF1=1.0` | The top-level IDF1 calculation measures detection matching rather than identity consistency. |

The unit defect follows directly from
`cv2.findHomography(image, field, cv2.RANSAC, reprojection_threshold_px)` in
`calibration.py`: OpenCV's inlier residual is measured in destination coordinates,
which here are yards. The reported errors likewise subtract field coordinates.
See [OpenCV's homography definition](https://docs.opencv.org/3.4.7/d9/d0c/group__calib3d.html).
Fix the fitting direction or explicitly change the unit contract; do not preserve the
current pixel label on yard residuals.

The two smallest probes can be reproduced from the repository root:

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import cv2
import numpy as np
from football_tracking.identity import match_tracklets
from football_tracking.calibration import Homography, ImagePoint, FieldPoint

scores = {(a, b): 0.9 for a in ('a', 'b') for b in ('c', 'd')}
print([(x.left_key, x.right_key, x.decision)
       for x in match_tracklets(['a', 'b'], ['c', 'd'], scores)])
image = np.array([(x, y) for x in (0, 100, 200)
                  for y in (0, 100, 200)], dtype=float)
field = image / 10
field[4] += (1.5, 0)
h = Homography.fit([ImagePoint(*p) for p in image],
                   [FieldPoint(*p) for p in field])
predicted = cv2.perspectiveTransform(image.reshape(-1, 1, 2),
                                   np.array(h.matrix)).reshape(-1, 2)
print(h.median_error_px, np.median(np.linalg.norm(predicted - field, axis=1)))
PY
```

### Additional source-inspected gaps

- `calibration.py`: no frame/time support, source-image provenance, withheld-landmark
  validation, or contact-point confidence. `is_valid` uses fitting residuals.
- `export.py:write_calibration_json`: any nonempty mapping is labeled valid without
  inspecting each homography's validity.
- `replay.py:PlayAlignment`: one frame/fps offset per shot; no replay speed or edit model,
  and no check that an event anchor belongs to its named shot.
- `cli.py`: resolves only the first two aligned shots having field samples; any accepted
  link sets global resolved/complete status. Shots with no field samples are filtered
  before unresolved reporting.
- `identity.py`: crop availability normally forces a team assignment; union-find has no
  component-level play, simultaneous-visibility, or contradictory-evidence checks.
- `evaluation.py`: independent best-overlap reference voting can attribute duplicate or
  mixed tracklets misleadingly; accepted unattributable pairs are absent from scoring.
- `cli.py`: run/config identity omits calibration, alignment and resolver-policy inputs;
  cache-hit proxy reporting relies on the absent live detector rather than cached
  provenance. These need regression coverage before evidence is promoted.

These are implementation findings, not measured estimates of their frequency in the
user's game. Existing tests passed for the audited modules, with the optional TrackEval
test skipped; passing those tests does not cover the new reproductions.

Validation command (exit 0):

```bash
UV_CACHE_DIR=.uv-cache uv run --extra dev python -m pytest \
  tests/test_calibration.py tests/test_identity.py tests/test_replay.py \
  tests/test_evaluation.py tests/test_cli.py -q
```

The initial command without the development extra could not locate pytest in the fresh
worktree environment; installing the declared development extra resolved it.

### Full-game asset inspection

`data/all-22-lions-rams.mp4` is present in the main checkout, but this worktree does not
have a `data` directory. Read-only ffprobe inspection reports 1920×1080,
306,151 video frames, 5107.618915 seconds of video, frame rate 19001/317 and time base
1/19001. The file size is 3,260,334,633 bytes. These are metadata values, not verified
full-decode counts.

A local ignored preview sheet at `artifacts/identity-research/full-game-preview.jpg`
contains twelve approximate seek previews at 30/35/40, 600/605/610,
1800/1805/1810 and 3600/3605/3610 seconds. Visual inspection shows sideline and end-zone
views, yard/hash markings, officials, goalpost obstruction and changing framing.
These previews support selecting calibration data; they do not establish exact shot
boundaries, matching play pairs, reviewed landmarks or timing anchors.

The full game can support development and held-out play calibration/identity evaluation.
It remains a single-game asset, so cross-game generalization requires additional games.
