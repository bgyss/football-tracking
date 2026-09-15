# Cross-shot identity: measured state and blocked gates

Commit: `db2475b` (branch `feat/cross-shot-identity`, tip at the time this document was written).

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
