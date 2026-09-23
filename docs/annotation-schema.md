# Reviewed football annotation schema

The repository uses version 1 JSON manifests for reviewed labels. Coordinates, frame
indices, and presentation timestamps stay in the original source-video coordinate
system. Inference output is a proposal; it becomes reference data only after a reviewer
has corrected it and explicitly marked it reviewed.

```json
{
  "schema_version": 1,
  "reviewed": true,
  "source": {
    "sha256": "...",
    "width": 1920,
    "height": 1080,
    "frame_count": 306151,
    "time_base": [1, 19001]
  },
  "shots": {
    "shot-0": {
      "start_frame": 1200,
      "end_frame": 1450,
      "play_id": "game-001-play-0042",
      "split": "development",
      "camera_label": "sideline"
    }
  },
  "annotations": [
    {
      "id": "shot-0-frame-1234-cvat-17",
      "shot_id": "shot-0",
      "source_frame": 1234,
      "pts": 123400,
      "bbox_xyxy_px": [100, 200, 130, 300],
      "label": "player",
      "track_id": "cvat-17",
      "global_id": "anon-player-0003",
      "team": "DET",
      "visibility": "partially_visible",
      "occluded": false,
      "jersey_readable": false,
      "review_status": "reviewed",
      "coordinate_space": "source",
      "reviewer": "reviewer-1",
      "revision": 1,
      "reviewed_at": "2026-09-15T00:00:00Z",
      "annotation_confidence": 1.0
    }
  ],
  "timing_events": [
    {
      "id": "play-0042-shot-0-snap",
      "shot_id": "shot-0",
      "source_frame": 1234,
      "pts": 123400,
      "event": "snap",
      "play_id": "game-001-play-0042",
      "correspondence_id": "snap",
      "play_time_s": 0.0,
      "review_status": "reviewed",
      "reviewer": "reviewer-1",
      "revision": 1,
      "reviewed_at": "2026-09-15T00:00:00Z",
      "annotation_confidence": 1.0
    }
  ],
  "landmarks": [
    {
      "id": "shot-0-frame-1234-yardline-20-near",
      "shot_id": "shot-0",
      "source_frame": 1234,
      "pts": 123400,
      "image_xy_px": [510, 440],
      "field_xy_yards": [20, 0],
      "landmark_id": "yardline:20:sideline:near",
      "role": "fit",
      "review_status": "reviewed",
      "reviewer": "reviewer-1",
      "revision": 1,
      "reviewed_at": "2026-09-15T00:00:00Z",
      "annotation_confidence": 1.0
    }
  ],
  "frame_labels": []
}
```

The `shots` map uses zero-based source frames with inclusive `start_frame` and exclusive
`end_frame`. Shot ranges cannot overlap. CVAT video XML does not carry shot boundary tags;
the export sidecar records the effective run shot segments and their source shot IDs.
Reviewers can correct those ranges in `provenance.json` before import. Source PTS values
remain separate from frame numbers and play time.

## Object labels and identity

Object `label` values are `player`, `official`, and `football`. Older version 1 manifests
that omit `label` are read as `player`. The MOT/reference exporter scores player boxes
only; officials and football remain available in the annotation manifest without being
treated as player tracks.

`track_id` is a sequence-local track ID and must be stable across frames of one shot.
`global_id` is an optional persistent anonymous ID linking a reviewed player across shots
of the same play. Keep these IDs anonymous (`anon-player-0003`), never a roster identity
or jersey number. Use `unknown` or `ambiguous` when a cross-shot identity is unresolved;
those values do not create a global identity link. Model-proposed `player_id` values are
stored separately as `proposal_player_id` and do not become reviewed `global_id` values.

The `team` attribute uses the teams configured for the dataset (the sample uses `DET`
and `LAR`) plus `unknown`. `visibility` is one of `visible`, `partially_visible`,
`occluded`, `out_of_frame`, or `unknown`; `occluded` is also carried as a boolean for
CVAT/MOT consumers that need the distinction. `jersey_readable` is a boolean. CVAT uses
`review_status` values `unreviewed`, `reviewed`, `accepted`, and `rejected`. Imported
records receive reviewed metadata only when the importer is explicitly invoked with
`--reviewed`, `--reviewer`, and `--reviewed-at`. Each visible CVAT shape must be marked
`reviewed`, `accepted`, or `rejected`; an `unreviewed` or missing status stops import, and
rejected shapes are omitted from the manifest and MOT reference. CVAT `outside` shapes
are omitted because they have no visible box to score.

## Timing anchors and calibration landmarks

`timing_events` record snap and corresponding action events with the original
`source_frame` and `pts`, a `play_id`, and optionally a `correspondence_id` shared across
replay views. `play_time_s` is a reviewed play-relative time; it never replaces or
rewrites the media PTS. In CVAT video XML, timing events are `timing_event` point tracks
with `event`, `play_id`, `correspondence_id`, `play_time_s`, and `source_pts` attributes.

Landmarks use `image_xy_px`, `field_xy_yards`, `source_frame`, and `pts`; `role` is `fit`
or `withheld`. Each keyframe needs at least four fit landmarks and one independent
withheld landmark to form a schema-v2 calibration timeline. A semantic `landmark_id`,
such as `yardline:20:hash:near`, is checked against its canonical field coordinate.
CVAT represents these as `calibration_landmark` point tracks with the field coordinates,
semantic ID, and fit/withheld role as attributes.

Field landmarks use a fixed orientation: x=0 is the west end line, x=120 is the east end
line, and y=0 is the near sideline. The field is 120 by 53 1/3 yards, with goal lines at
x=10 and x=110 and hash rows at y=70.75/3 and y=53 1/3−70.75/3. Repeated markings need
the side and field orientation so a mirrored fit cannot pass by appearance alone.

Calibration exports retain the RANSAC fit inlier/outlier indices. Local projection
sensitivity in yards per declared pixel uncertainty is diagnostic until the
contact-quality policy is frozen and reviewed on player labels.

## CVAT bridge and provenance

[`scripts/export_cvat.py`](../scripts/export_cvat.py) reads `observations.csv` from one
run and writes CVAT for video 1.1 XML plus `provenance.json`. The XML uses the complete
source video frame space: CVAT `frame` is the original zero-based `source_frame`, and
each proposal carries `source_pts`, `detection_score`, team evidence, `source_run_id`,
`source_tracklet_id`, and `proposal_player_id`. `anonymous_id` starts as `unknown` and
`review_status` starts as `unreviewed`. The sidecar records the source hash, run and
tracker configuration, shot ranges, the observations CSV hash, and the original rows for
each proposal, including the source shot ID when a reviewed alignment assigned another
effective shot name. CVAT video XML stores snap/play-time anchors as point tracks; shot
boundaries stay in the sidecar. Keep the sidecar with the XML through review. See the [CVAT video XML
format](https://docs.cvat.ai/docs/manual/advanced/formats/format-cvat/).

[`scripts/import_cvat.py`](../scripts/import_cvat.py) checks that the source video hash
and dimensions match the sidecar, resolves each PTS from the verified source frame index,
preserves frame numbers, and writes both
`annotations.json` and `mot-reference.json`. A box's original inference row is retained
under `inference_provenance` when its shot, frame, and source tracklet still match. New
human-added boxes have no fabricated inference provenance. Only reviewed `player`
objects enter the MOT-style reference; officials, football, timing events, and landmarks
stay in the annotation manifest.

The example above illustrates the reviewed form. A generated preannotation is always
`reviewed: false`; do not use it as ground truth. Reviewers must confirm source frames,
shot intervals, play grouping, split, landmark semantics, visibility, and identity
ambiguity before importing it as reviewed data.

## Evaluation and calibration notes

Keep these separate:

- fitting landmarks and withheld landmarks for calibration;
- visible-player boxes and contact-point confidence for position evaluation;
- within-shot track identity and cross-shot global identity;
- snap/corresponding events for play-time alignment.

Use `frame_labels` for reviewed empty or ignored frames. An empty `objects` list with
`labeled: true` means the reviewer inspected the frame and found no evaluable objects;
`ignore: true` excludes that frame under the declared evaluation policy. Omitting a frame
does not silently turn it into negative ground truth.

Player records may include `ground_contact_xy_yards` and an explicit
`ground_contact_confidence`. Contacts below the frozen evaluator confidence threshold are
reported as excluded rather than counted as position failures.

The review pack produced by `scripts/build_identity_review_pack.py` is intentionally
`reviewed: false`. Reviewers must confirm source frame, PTS, shot interval, play grouping,
split, landmark semantics, visibility/occlusion, and identity ambiguity before changing
that state. It may include Hough field-line proposals, but those are hints and must be
replaced by semantic, human-reviewed landmarks. Unknown or ambiguous players remain
explicit and are excluded only under the declared evaluation policy.

`scripts/build_annotation_manifest.py` copies the pack's source metadata and frame/PTS
records into a manifest template after the reviewer declares shot ranges and camera labels.
It keeps `reviewed: false`; reviewers still must add and approve all boxes, landmarks,
timing events, contacts, and identities.

For moving All-22 cameras, `calibration_timeline.py` can estimate a field-only
inter-frame transform when supplied a static-field mask. The transform is composed as
`H_current = H_keyframe @ inverse(G_current_from_keyframe)` and is discarded when fewer
than four robust correspondences remain. This motion estimate is a propagation proposal;
absolute reviewed landmarks and withheld validation remain the authority for identity.
`propagate_calibration_timeline` applies reviewed motion steps only within a declared PTS
gap and ends support when that gap is exceeded; it never extrapolates through an unsupported
camera move.

Once the manifest contains reviewed landmark records, call
`timeline_from_landmark_records(manifest.landmarks)` to fit the schema-v2 timeline. Each
keyframe needs at least four `fit` records and one independent `withheld` record; intervals
are bounded by the next reviewed keyframe and are never extrapolated. The generated
timeline must carry the input `source_sha256`; the CLI refuses source-hashed mismatches
and does not permit legacy static files to create cross-shot joins.
