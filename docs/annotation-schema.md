# Reviewed calibration and identity annotation schema

The repository accepts version 1 JSON manifests only when the source hash and source
coordinates match the video being evaluated. A model proposal is never ground truth.

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
      "id": "shot-0-frame-1234-player-17",
      "shot_id": "shot-0",
      "source_frame": 1234,
      "pts": 123400,
      "bbox_xyxy_px": [100, 200, 130, 300],
      "track_id": "player-17",
      "team": "DET",
      "visibility": "visible",
      "ground_contact": "evaluable",
      "review_status": "reviewed",
      "coordinate_space": "source",
      "reviewer": "reviewer-1",
      "revision": 1,
      "reviewed_at": "2026-09-15T00:00:00Z",
      "annotation_confidence": 1.0
    }
  ]
}
```

Landmark records use the same manifest with `image_xy_px`, `field_xy_yards`, `role` (`fit`
or `withheld`), `source_frame`, `pts`, and an optional semantic `landmark_id` such as
`yardline:20:hash:near`. When a semantic ID is present, its canonical field coordinate is
checked before fitting, which catches mirrored or mislabeled field orientations.

The source image is 1920×1080 in the supplied All-22 file. `source_frame` and `pts` are
kept in the original media coordinate system. Crop or resized review images must be
converted with `source_bbox_from_crop` before a record is marked reviewed.

Field landmarks use a fixed orientation: x=0 is the west end line, x=120 is the east end
line, and y=0 is the near sideline. The field is 120 by 53 1/3 yards, with goal lines at
x=10 and x=110 and hash rows at y=70.75/3 and y=53 1/3−70.75/3. Use semantic IDs such as
`yardline:20:hash:near` and `goal_line:west:sideline:near`; repeated markings require
the side and field orientation so a mirrored fit cannot pass by appearance alone.

Calibration exports retain the RANSAC fit inlier/outlier indices. A caller can also ask for
local projection sensitivity in yards per declared pixel uncertainty; this is diagnostic
until the contact-quality policy is frozen and reviewed on player labels.

Keep these labels separate:

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
