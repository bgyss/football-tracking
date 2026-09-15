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
      "split": "development"
    }
  },
  "annotations": [
    {
      "id": "shot-0-frame-1234-player-17",
      "shot_id": "shot-0",
      "source_frame": 1234,
      "bbox_xyxy_px": [100, 200, 130, 300],
      "track_id": "player-17",
      "team": "DET",
      "visibility": "visible",
      "ground_contact": "evaluable",
      "review_status": "reviewed",
      "coordinate_space": "source"
    }
  ]
}
```

The source image is 1920×1080 in the supplied All-22 file. `source_frame` and `pts` are
kept in the original media coordinate system. Crop or resized review images must be
converted with `source_bbox_from_crop` before a record is marked reviewed.

Field landmarks use a fixed orientation: x=0 is the west end line, x=120 is the east end
line, and y=0 is the near sideline. The field is 120 by 53 1/3 yards, with goal lines at
x=10 and x=110 and hash rows at y=70.75/3 and y=53 1/3−70.75/3. Use semantic IDs such as
`yardline:20:hash:near` and `goal_line:west:sideline:near`; repeated markings require
the side and field orientation so a mirrored fit cannot pass by appearance alone.

Keep these labels separate:

- fitting landmarks and withheld landmarks for calibration;
- visible-player boxes and contact-point confidence for position evaluation;
- within-shot track identity and cross-shot global identity;
- snap/corresponding events for play-time alignment.

The review pack produced by `scripts/build_identity_review_pack.py` is intentionally
`reviewed: false`. Reviewers must confirm source frame, PTS, shot interval, play grouping,
split, landmark semantics, visibility/occlusion, and identity ambiguity before changing
that state. Unknown or ambiguous players remain explicit and are excluded only under the
declared evaluation policy.

For moving All-22 cameras, `calibration_timeline.py` can estimate a field-only
inter-frame transform when supplied a static-field mask. The transform is composed as
`H_current = H_keyframe @ inverse(G_current_from_keyframe)` and is discarded when fewer
than four robust correspondences remain. This motion estimate is a propagation proposal;
absolute reviewed landmarks and withheld validation remain the authority for identity.
