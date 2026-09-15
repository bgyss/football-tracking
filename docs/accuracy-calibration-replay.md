# Improving Football Tracking Accuracy, Yard Calibration, and Replay Identity

This guide explains how to take the current tracking pipeline from an integration baseline to a football-accurate system with field coordinates and cross-view replay identity. It complements the broader [system design](system-design.md) and the measurable [evaluation plan](evaluation-plan.md).

For the practical data loop—authorized footage, initial-track preannotations, CVAT review, optional SAM 2 propagation, and COCO/MOT exports—see [Local annotation and data generation](annotation-and-data-generation.md).

The three problems are related, but they need separate modules and evidence:

1. **Football accuracy** determines whether players are detected, tracked, and assigned to teams reliably.
2. **Yard-space calibration** maps image observations into a fixed coordinate system on the field.
3. **Replay identity** decides whether tracklets from different camera views belong to the same player in the same play.

Keeping these seams separate prevents a plausible-looking overlay from being mistaken for a validated football measurement.

## Football accuracy

The current detector interface is in `src/football_tracking/detector.py`. `RFDETRDetector` converts OpenCV BGR frames to RGB, preserves original pixel coordinates, and normalizes RF-DETR output into `Detection` records. The current full-clip run used generic RF-DETR Small weights, so its 17,222 observations and 59 tracklets are integration evidence only.

### Fine-tune the detector

Create a football dataset in Roboflow with at least `player`, `official`, and `football` classes. Include:

- wide All-22 formations where players are small;
- line-of-scrimmage overlap, blocking, piles, tackles, and motion blur;
- partially visible players and players near the boundary;
- officials, coaches, sideline personnel, cables, logos, and turf as hard negatives;
- both sideline and end-zone viewpoints.

Split by game or play. Never put adjacent frames, or two replay views of one play, into different train and test partitions. Ordinary detection boxes are not persistent identity labels, so annotate track IDs separately for evaluation.

Start with RF-DETR Small and compare Medium only after measuring distant-player recall. If small players are missed, evaluate larger inference resolution and overlapping tiles before increasing tracker complexity. Deduplicate tile predictions in original-image coordinates before sending them to the tracker.

Use a football-fine-tuned checkpoint with the command-line pipeline:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run \
  --input data/all-22-lions-rams-sample.mp4 \
  --output artifacts/football-botsort \
  --detector rfdetr \
  --detector-checkpoint /path/to/football-rfdetr.pth \
  --model-size small \
  --tracker botsort \
  --manual-cut 712
```

### Tune tracking and teams

`src/football_tracking/tracking.py` provides Roboflow ByteTrack and BoT-SORT adapters. Compare both with the same cached detections. BoT-SORT's camera-motion compensation should be tested with identical settings enabled and disabled; a ByteTrack-versus-BoT-SORT comparison alone does not isolate camera compensation.

Tune detector confidence, association thresholds, `lost_track_buffer`, and the minimum consecutive-frame rules together. Keep low-confidence detections available for recovery instead of filtering them all at a high cutoff. Reset the tracker at every camera cut.

Roboflow's maintained BoT-SORT adapter does not provide appearance ReID. Add a football-specific appearance module above it that combines several crops from each tracklet, jersey-number evidence, team evidence, and motion. A general person-ReID model is only an experiment because same-team uniforms and helmets are visually similar.

The current `team_feature_from_crop` implementation in `src/football_tracking/identity.py` is a trimmed torso RGB feature. It is a useful baseline, but color clustering only separates uniform groups. For stronger team labels, use masked torso pixels or a learned crop classifier, aggregate over time, and retain `unknown` for weak crops and officials. Use `--team-prototypes` when the team-to-cluster mapping is known.

Evaluate visible-player precision and recall, HOTA, IDF1, fragmentation, and ID switches. The targets in `docs/evaluation-plan.md` are proposed gates: 0.95 detector precision and recall, IDF1 of at least 0.90, and at most one within-shot ID switch on the development clip. They are not satisfied by the current generic-checkpoint run.

## Yard-space calibration

The calibration interface is in `src/football_tracking/calibration.py`. `Homography.fit` estimates a robust image-to-field transform and `project_observation` projects the bottom center of a player box. `load_calibrations` supports a shared transform or separate transforms keyed by `shot-0`, `shot-1`, and so on.

Use a fixed field coordinate system for the whole analysis: x runs from 0 to 120 yards along the field, and y runs from 0 to 53.333 yards across it. Record the direction explicitly. Do not redefine the direction based on possession or camera orientation.

For each shot, label at least 6–8 well-distributed intersections of yard lines, hash marks, sidelines, and other fixed field markings. Avoid four points on one line. Keep additional landmarks withheld so calibration quality is measured on points that were not used for fitting.

```json
{
  "shots": {
    "shot-0": {
      "image_points": [[100, 200], [900, 180], [980, 650], [40, 680]],
      "field_points": [[20, 10], [60, 10], [60, 45], [20, 45]]
    },
    "shot-1": {
      "image_points": [[120, 160], [1160, 160], [1050, 680], [180, 680]],
      "field_points": [[40, 5], [80, 5], [80, 48], [40, 48]]
    }
  }
}
```

The coordinates above are illustrative; they must be replaced with landmarks measured in the actual frames. Run with `--calibration data/calibration.json`. The current sample has no calibration file, which is why `trajectories.csv` contains image-space contact points while its yard columns are empty and `field-view.png` contains only the field grid.

A static homography per shot is sufficient for a fixed camera. All-22 footage often pans and zooms, so the next calibration module should store time-keyed homographies. Refit from field features at keyframes, interpolate only between valid fits, and invalidate positions when reprojection error rises. Camera-motion compensation used by BoT-SORT and metric field calibration solve different problems and need separate quality checks.

Bottom-center box contact is approximate for crouching, airborne, tackled, or truncated players. Add pose or segmentation foot points when the withheld ground-contact error shows that this dominates. Keep observed, predicted, interpolated, and invalid positions distinct; never fill a long occlusion as measured movement.

The calibration gates are a median withheld-landmark error of at most 1 yard, a 95th-percentile error of at most 2 yards, and valid ground-contact positions for at least 90% of evaluable observations.

## Cross-view replay identity

The sample changes camera perspective at frame 712 (approximately 11.88 seconds), and the second view appears to replay the same play. `shot_id`, `play_id`, and `player_id` must remain separate:

- `shot_id` identifies image-space tracking state;
- `play_id` identifies the underlying football play;
- `player_id` identifies an anonymous player across compatible tracklets.

When `--play-alignment` is supplied, the CLI now builds candidate cross-shot links from `replay.cross_shot_candidate_scores` over calibrated field positions and passes them through `identity.match_tracklets` before calling `stable_anonymous_ids`; every decision — resolved, abstained, or not attempted — is recorded in `identity-links.json`. Without `--play-alignment`, or without calibrated field positions for at least two aligned shots, the resolver abstains and `stable_anonymous_ids` still receives an empty link list, so each tracklet keeps its own deterministic, shot-local ID and no tracklets are joined across shots. `review.json` records this as unresolved. Abstention on uncalibrated footage is deliberate: image-coordinate proximity across the cut is meaningless. See [docs/evidence/cross-shot-identity.md](evidence/cross-shot-identity.md) for the measured status on the current assets — cross-shot identity has not been demonstrated to resolve correctly on real footage; only the plumbing has been exercised.

Implement replay matching in a new module or as a deeper layer above `identity.py`:

1. Annotate snap/action anchors and assign a common `play_time_s` to each shot. Keep media timestamps unchanged.
2. Confirm the second view is a replay using formation, field markings, and action timing. Do not append the replay to the first trajectory.
3. Build a candidate score from team agreement, calibrated field positions at aligned times, trajectory shape, jersey-number consensus, and football-specific appearance embeddings.
4. Pass candidate scores into the existing `match_tracklets` function. It already supports one-to-one assignment, score thresholds, and abstention margins.
5. Pass accepted `IdentityLink` records into `stable_anonymous_ids`. Preserve rejected and insufficient-evidence links with their supporting crop or frame IDs.
6. Reject team conflicts, impossible field motion, duplicate simultaneous identities, and weak margins. Keep unresolved tracklets separate.

GPT-6 Astra can assist only with selected ambiguous crop comparisons. Supply numbered crops and candidate IDs, request `same`, `different`, or `insufficient_evidence`, validate the structured response locally, and keep the local constraints authoritative. The current pipeline makes no Astra requests.

Measure cross-view precision and coverage independently from within-shot IDF1. The proposed cross-view gate is zero false merges with at least 0.80 coverage of human-resolvable shared players. A correct replay merge represents one play with two observations; it must not double-count distance, events, or player participation.

## Recommended order

1. Build the football training set and fine-tune RF-DETR.
2. Annotate development and held-out identity sequences, then tune ByteTrack and BoT-SORT on the same detection cache.
3. Label shot-specific calibration landmarks and add time-varying transforms when camera motion requires them.
4. Add play-time alignment and constrained cross-view identity joins.
5. Add Astra only after the local resolver has a measured ambiguity queue and a clear cost/quality comparison.

The current [evaluation plan](evaluation-plan.md) should be updated with the resulting annotations, model versions, thresholds, and failure clips before making football-wide accuracy claims.
