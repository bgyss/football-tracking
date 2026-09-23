# Human review of the supplied All-22 footage

This is the labeling procedure for `data/all-22-lions-rams.mp4`. The goal is to
produce evidence that can support field calibration, replay timing, within-shot
tracking, and cross-shot identity. A detector proposal, Hough line proposal, or
tracker ID is only a starting point; it is never ground truth.

## 1. Register the source and choose a review batch

Keep the original video unchanged. The current source is 1920×1080, 306,151
frames, with time base `[1, 19001]`. The review artifacts must retain the source
frame number and source PTS for every label.

The existing scouting pack is at
`artifacts/full-game-calibration-review-pack/review-pack.json`, with a contact
sheet at `artifacts/full-game-calibration-review-pack/contact-sheet.jpg`. It is
explicitly `reviewed: false`. It contains twelve exact source frames and line
proposals for scouting; it is not a complete player or identity annotation.

For a first pass, select one play that has at least two views (for example a
sideline view and an end-zone or replay view), then review a contiguous window
from before the snap through the end of the play. Add more plays only after the
first one passes the checks below. Record each shot as:

| Field | Meaning |
| --- | --- |
| `shot_id` | Unique source-video shot interval |
| `start_frame`, `end_frame` | Source frame range; end is exclusive |
| `camera_label` | `sideline`, `endzone`, `replay`, or another explicit label |
| `play_id` | Same value for shots showing the same play |
| `split` | `development`, `validation`, or `test`; keep a play in one split |

Confirm every cut by scrubbing a few frames on both sides. Do not infer a cut
from a model score alone. The current inventory script is a scouting aid and
must also be reviewed before its boundaries are used.

## 2. Review calibration landmarks

Open the original-resolution frame, not only the contact-sheet thumbnail. At
each camera-motion keyframe, mark at least four `fit` landmarks and at least one
independent `withheld` landmark. More than the minimum is preferable when the
field is visible.

Use precise, repeatable field features such as yard-line/hash intersections,
sidelines, end lines, and goal lines. Click the same semantic point in the image
and record its canonical field coordinate. Use semantic IDs where possible,
such as `yardline:20:hash:near` or
`goal_line:west:sideline:near`; these catch mirrored field orientations.

The canonical field coordinate system is fixed: `x=0` is the west end line,
`x=120` is the east end line, and `y=0` is the near sideline. Do not flip the
orientation to make a fit look better. Hough line and intersection proposals in
the pack are hints. `scripts/export_cvat.py --review-pack ...` can load the
intersections as point tracks; assign a semantic `landmark_id` and role before
importing them into the calibration manifest.

Add a new keyframe after a pan, zoom, or other camera move changes the
projection. A propagated calibration may only cover the declared supported PTS
interval; leave an unsupported gap rather than extrapolating through a cut or
large camera move.

The calibration gate is independent of the fit points: withheld median error must
be at most 1 yard and withheld p95 error at most 2 yards. If it fails, correct the
landmarks or shorten the supported interval; do not simply mark the estimate
valid.

## 3. Review play timing

For each multi-view play, mark the same visible events in every shot using source
PTS. A useful event set might include `pre_snap`, `snap`, and `ball_release` (use
only events that are actually visible and consistently defined). Use at least two
correspondences per shot to fit the play-time map and hold out a separate event
per shot for validation. The fit and validation event labels must be disjoint.

Timing labels should be recorded in a reviewed play-alignment JSON with:

- `reviewed: true`, `play_id`, and the source SHA-256;
- an anchor and source frame/PTS for every shot;
- the reviewed `shot_ranges`;
- common-event `correspondences` with `shot_id`, `source_pts`, `play_time_s`, and
  `event`;
- held-out `validation_correspondences` and the resulting `timing_report`.

Never align two shots by wall-clock position in the file when the footage contains
replay freezes, edits, or slow motion. The evaluator refuses to extrapolate
outside the reviewed correspondence support.

For a bounded candidate window, `scripts/propose_timing_events.py` can emit
source-frame/PTS motion-burst proposals. It does not classify a burst as `snap`
or `ball_release`; review the window and assign the shared event names manually.

## 4. Review players and within-shot tracks

Use a local video annotation tool such as CVAT Community for the contiguous
window. The repository's `scripts/export_cvat.py` writes proposal tracks plus a
source-hashed `task-frame-map.json` containing the source frame, integer source
PTS, time base, and any crop transform. `scripts/import_cvat.py` verifies that
map against the original video and converts corrected boxes back into source
coordinates. Keep the raw `observations.csv` cache immutable.

The CVAT export can also include `ground_contact` point proposals and
`field_landmark` intersection proposals. Correct contact points in original
source coordinates, set `ground_contact_confidence`, and assign each field point
its semantic ID and `fit` or `withheld` role. The importer stores unresolved
point proposals separately and keeps the manifest unreviewed until they are
resolved.

For every inspected frame:

1. Draw each evaluable player box in original source pixels.
2. Give the object a shot-local `track_id`; split it when identity becomes
   ambiguous instead of stretching one ID across an occlusion or cut.
3. Label team as `DET`, `LAR`, or `unknown`. Use `unknown` when the jersey or
   evidence is not readable.
4. Mark visibility/occlusion and use an `outside` interval when the player leaves
   the image. Do not drag a box through a pile or an out-of-view interval.
5. Record `frame_labels` for inspected empty frames and for frames intentionally
   ignored by the evaluation policy. An omitted frame is not an implicit negative
   label.
6. Where position evaluation is required, mark the player's ground-contact point
   in canonical field yards and a `ground_contact_confidence`. Do not substitute
   the box center for a foot/contact point. Leave the contact unevaluable or low
   confidence when the feet are hidden.

Review densely around camera cuts, crossings, blocking, tackles, piles, and
recovery from occlusion. Interpolation is a labor-saving proposal and must be
corrected at those events.

## 5. Review cross-shot identity conservatively

Cross-shot identity is play-scoped. Link a shot-local object to a `global_id`
only when the evidence supports the same player across the two shots: shared
play timing, calibrated field position and motion, compatible team evidence, and
visible jersey/body continuity. A readable number is strong evidence; color or
nearby position alone is not.

Use explicit `unknown`, `ambiguous`, or `unresolved` values when the footage does
not support a decision. Do not force a one-to-one assignment to make every
tracklet link. In particular, do not link players from unrelated plays, do not
carry an ID across a replay cut without reviewed timing, and do not treat a
detector `tracklet_id` as a global player identity.

Have a second reviewer inspect every accepted cross-shot link and every rejected
or ambiguous link. Record the reviewer, revision, timestamp, and confidence for
each reviewed object, landmark, and frame label.

Use `scripts/build_identity_review_queue.py` to order accepted links, close
alternatives, rejected edges, and unmatched tracklets for manual inspection. Its
optional contact sheet is only a navigation aid; the review item includes the
original source frame and PTS for each sample. `scripts/propose_jersey_reads.py`
can produce unreviewed OCR candidates for sufficiently large crops. The run
resolver only uses explicitly reviewed reliable number conflicts as a hard
constraint; automatic OCR or appearance cues only change review ordering.

## 6. Build and validate the repository artifacts

Create an unreviewed manifest skeleton after the shot table is confirmed:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/build_annotation_manifest.py \
  --review-pack artifacts/full-game-calibration-review-pack/review-pack.json \
  --output artifacts/full-game-calibration-review-pack/manifest-template.json \
  --shot shot-0:START_FRAME:END_FRAME:sideline:play-0042:development
```

Import corrected CVAT tracks into the template with `scripts/import_cvat.py`.
The default import remains unreviewed. To promote a completed, checked batch,
pass `--mark-reviewed`, `--reviewer`, `--revision`, `--reviewed-at`, and
`--annotation-confidence`; the importer validates those fields and the source
mapping before writing the manifest. Field semantics, ground-contact confidence,
play links, and global identity still require explicit human review.

Then create the derived artifacts:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_calibration_timeline.py \
  --annotations /path/to/reviewed-manifest.json \
  --source data/all-22-lions-rams.mp4 \
  --output artifacts/calibration-timeline.json

UV_CACHE_DIR=.uv-cache uv run python scripts/build_reviewed_reference.py \
  --annotations /path/to/reviewed-manifest.json \
  --source data/all-22-lions-rams.mp4 \
  --calibration artifacts/calibration-timeline.json \
  --output artifacts/reviewed-reference.json

UV_CACHE_DIR=.uv-cache uv run python scripts/check_identity_readiness.py \
  --source data/all-22-lions-rams.mp4 \
  --annotations /path/to/reviewed-manifest.json \
  --calibration artifacts/calibration-timeline.json \
  --play-alignment /path/to/reviewed-play-alignments.json \
  --reviewed-reference artifacts/reviewed-reference.json
```

The final command must report `ready_for_evaluation`. A `not_ready` result is a
useful diagnosis: it identifies which source-hashed calibration, timing, or
reviewed-reference gate is still missing. Do not report identity accuracy until
the real footage has passed this readiness check and the held-out evaluation
windows have been sealed.
