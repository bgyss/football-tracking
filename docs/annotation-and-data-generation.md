# Local annotation and data generation workflow

This guide turns football video into reviewed training data for this repository without using Roboflow-hosted services. The recommended stack is:

- **CVAT Community** for video tracks, keyframes, interpolation, correction, and review;
- **this repository** for RF-DETR inference, cached detections, BoT-SORT/ByteTrack proposals, and export;
- **SAM 2** only when a difficult occlusion needs a mask-assisted proposal;
- **FiftyOne** for dataset curation and active-learning frame selection.

The initial track is a preannotation. It is never accepted as ground truth without human review.

## What the Roboflow football tutorial contributes

The [Roboflow American-football player-tracking tutorial](https://blog.roboflow.com/american-football-player-tracker/) is a useful source-specific recipe, published April 24, 2026. It uses RF-DETR Small for detection, ByteTrack for short-term association, and bounding-box, label, and trace overlays. It also recommends a domain-specific American-football dataset, labeling players by team or role, and adding examples from different lighting and jersey patterns.

The post reports a 70/20/10 train/validation/test split, horizontal flips, saturation and brightness changes of approximately ±25%, and 100-pixel motion blur. It reports mAP@50 of 74.8%, precision of 90.8%, recall of 71.5%, and an F1 score of 77.9% at a 59% confidence threshold. Those numbers are the tutorial author's results for its own dataset and configuration. They are not measurements of `all-22-lions-rams-sample.mp4`, and they do not establish persistent identity, cross-view replay matching, or yard-space accuracy.

The post also recommends SAHI-style slicing when players are small in wide shots. That is relevant to All-22 footage, but tiled inference must deduplicate boxes in full-frame coordinates before tracking. Its claim that ByteTrack maintains identity through brief occlusion is a useful motivation for a baseline, not proof that a football pile or replay cut will preserve a player ID.

The tutorial's remaining steps use Roboflow Universe, Annotate, Train, Workflows, and Production Metrics Explorer. This project intentionally replaces those hosted steps with local video decoding, CVAT review, local RF-DETR checkpoints, the Roboflow `trackers` package, and repository-owned metrics. The post names the model identifier `nfl-detection-1500-jdrgz/4`; treat it as a discovery lead only. Checkpoint access, class mapping, dataset provenance, and redistribution terms must be confirmed independently before use.

## Recommended open-source toolchain

### CVAT Community: primary annotation tool

Use [CVAT Community](https://github.com/cvat-ai/cvat) locally. Its video Track mode automatically interpolates boxes between keyframes and supports keyframe edits, `Outside` states, track splitting, and track merging. Its native track format stores per-frame boxes with fields such as `occluded`, `outside`, and `keyframe`; this maps directly to the observations produced by this repository. [CVAT Track mode](https://docs.cvat.ai/docs/manual/advanced/track-mode-advanced/), [CVAT format](https://docs.cvat.ai/docs/manual/advanced/formats/format-cvat/)

CVAT Community is MIT-licensed, self-hostable, and keeps the video in the local annotation environment. This repository now exports and imports CVAT video XML using a source-hashed `task-frame-map.json` sidecar. The sidecar maps each CVAT task frame to the original zero-based source frame and integer source PTS, and records any crop and resize transform. CVAT task-local frame numbering alone is never treated as source numbering. Its automatic-annotation interface can also run a custom tracking function under your control. [CVAT auto-annotation API](https://docs.cvat.ai/docs/api_sdk/sdk/auto-annotation/)

### SAM 2: targeted propagation and mask refinement

[SAM 2](https://ai.meta.com/research/sam2/) accepts a box, click, or mask prompt and propagates an object through video while allowing corrections on later frames. Use it for a player that is hard to box during a short occlusion or pile. Convert the reviewed mask back to a detector box if the RF-DETR training target is a box. Do not use a mask propagation result as evidence that the same jersey belongs to the same player across camera views.

### FiftyOne: curation and active learning

[FiftyOne](https://docs.voxel51.com/integrations/cvat.html) is useful for browsing clips, finding difficult or duplicate samples, and sending selected video tasks to CVAT. It can import video tracks and preserve track indices and keyframe information. It is a curation layer, not a replacement for CVAT's video editor.

### Alternatives

[Label Studio](https://github.com/HumanSignal/label-studio) supports video rectangles, and its open-source ML backend can generate video tracking predictions with BoT-SORT or ByteTrack. It is a reasonable alternative if that UI is already familiar, but its native video output would need a separate importer/exporter for this repository's `observations.csv` and `detections.jsonl`. [Video tracking template](https://labelstud.io/templates/video_object_detector.html), [ML backend](https://github.com/HumanSignal/label-studio-ml-backend)

[Autodistill](https://github.com/autodistill/autodistill) can use a foundation model to create seed labels for still frames. It is useful for selecting initial frames, but it is not the main tool for reviewing persistent football tracks.

## End-to-end local workflow

### 1. Acquire and register footage

Use `yt-dlp` only for videos you are authorized to download and use. Store a manifest containing the source URL, uploader, date, game or practice context, original resolution, downloaded hash, and permission or license notes. YouTube's Terms restrict downloading or using content unless authorized by the service, the rights holder, or applicable law; the downloader's software license does not change those rights. [yt-dlp](https://github.com/yt-dlp/yt-dlp), [YouTube Terms](https://au.youtube.com/t/terms)

Preserve the original frame rate and resolution. Do not train from a platform preview if a higher-quality authorized source is available. Split long videos into play or shot clips only after recording the relationship to the source video.

### 2. Generate local preannotations

Run the repository pipeline with a football checkpoint when one exists:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run \
  --input data/game-001.mp4 \
  --output artifacts/game-001 \
  --detector rfdetr \
  --detector-checkpoint /path/to/football-rfdetr.pth \
  --model-size small \
  --tracker botsort \
  --manual-cut 712
```

The run produces `detections.jsonl`, `observations.csv`, `observations.parquet`, `trajectories.csv`, and `annotated.mp4`. Export one shot or bounded review interval to CVAT:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/export_cvat.py \
  --observations artifacts/game-001/observations.csv \
  --source data/game-001.mp4 \
  --shot-id shot-12 --start-frame 12000 --end-frame 13350 \
  --review-pack artifacts/game-001/review-pack.json \
  --frame-map-output artifacts/game-001/shot-12-task-frame-map.json \
  --output artifacts/game-001/shot-12-cvat-bundle.zip \
  --cvat-xml-output artifacts/game-001/shot-12-annotations.xml
```

Import the standalone XML into a local CVAT video task created from that exact source interval with frame step 1. The ZIP bundle is for repository round-tripping; it includes both XML and the sidecar. If the task uses a crop or resize, pass the matching `--crop x1,y1,x2,y2` and `--task-size WIDTHxHEIGHT` when exporting, and prepare the CVAT media with the same transform. The task map binds every local frame to source PTS and maps edited boxes and points back to original 1920×1080 coordinates. The exporter breaks proposal tracks across long unobserved gaps and marks all boxes and track attributes as proposals. With `--review-pack`, Hough-line intersection hints are exported as unreviewed `field_landmark` point tracks; reviewers assign each semantic landmark ID and `fit` or `withheld` role before those points enter calibration.

After human review, import the corrected CVAT XML plus its sidecar into the repository manifest:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/import_cvat.py \
  --input artifacts/game-001/shot-12-reviewed.xml \
  --frame-map artifacts/game-001/shot-12-task-frame-map.json \
  --source data/game-001.mp4 \
  --shot shot-12:12000:13350:sideline:play-0042:development \
  --output artifacts/game-001/shot-12-annotations.json \
  --mark-reviewed \
  --reviewer reviewer-1 \
  --revision 1 \
  --reviewed-at 2026-09-23T12:00:00Z \
  --annotation-confidence 1.0 \
  --mot-reference-output artifacts/game-001/shot-12-mot-reference.json
```

The importer verifies source hash, dimensions, frame count, time base, exact PTS, and crop mapping. It emits `reviewed: false` by default. Use `--mark-reviewed` and reviewer metadata only after checking every imported shape and frame tag in CVAT; each visible shape must be `reviewed` or `accepted`, while `rejected` shapes are omitted. `--mot-reference-output` writes a player-only MOT reference and requires a fully reviewed manifest. A proposal import cannot become evaluation truth by itself.

The exporter also creates unreviewed `ground_contact` point tracks at each proposal box's bottom center. In CVAT, correct each point to the visible ground contact and set a confidence. Pass the valid calibration timeline to `scripts/build_reviewed_reference.py --calibration` to project reviewed source pixels into field yards.

Only enter a `global_id` after the second reviewer checks the candidate. In the player track's attributes, set `cross_shot_review_status=approved`, the agreed `global_id`, and the second reviewer's name, timestamp, revision, and confidence. Use `ambiguous` or `rejected` with no meaningful `global_id` when the evidence does not support a link. The importer refuses to promote a reviewed global identity without the independent review metadata.

Use `tracklet_id` as the proposal identity because the current `player_id` is deterministic per tracklet and cross-view replay identity is intentionally unresolved.

The converter preserves original frame numbers and boxes, attaches detector score and team suggestions as attributes, and emits observed boxes as keyframes. It does not silently discard detections or rewrite the raw cache.

### Source-addressed identity cues and review queues

Tesseract can propose jersey readings from crops that are large enough to inspect. OCR output stays unreviewed, retains source frame/PTS and the exact crop box, and is tied to the source video and base detector/tracker analysis hash:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/propose_jersey_reads.py \
  --source data/game-001.mp4 \
  --observations artifacts/game-001/observations.csv \
  --analysis-config artifacts/game-001/analysis-config.json \
  --output artifacts/game-001/jersey-cues.json
```

Use the cue file on a matching rerun with `--identity-cues`. The pipeline verifies each cue's tracklet, frame, PTS, and box against the current observations. Unreviewed jersey and appearance cues only change review ranking. A jersey conflict becomes a hard candidate rejection only after a reliable jersey value is explicitly marked reviewed or accepted. These cues do not assign `global_id` values by themselves.

Create a side-by-side identity review queue from a run:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/build_identity_review_queue.py \
  --identity-links artifacts/game-001/identity-links.json \
  --observations artifacts/game-001/observations.csv \
  --source data/game-001.mp4 \
  --output artifacts/game-001/identity-review-queue.json \
  --contact-sheet artifacts/game-001/identity-review-contact-sheet.jpg
```

The queue orders accepted links, close alternatives, rejected edges, and unmatched tracklets for inspection. Every sample includes source frame, source PTS, box, team, calibration status, and aligned play time where available. The contact sheet is an aid; the JSON source coordinates and original video remain authoritative.

`scripts/build_play_inventory.py` emits overlapping adjacent-shot play-window proposals alongside candidate cuts. These windows only prioritize review; confirm every camera transition and play grouping manually.

For a bounded source window, propose motion-burst frames to inspect for timing events and candidate camera-motion keyframes:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/propose_timing_events.py \
  --source data/game-001.mp4 --start-frame 12000 --end-frame 13350 \
  --output artifacts/game-001/shot-12-event-proposals.json
```

These proposals only identify motion changes or possible camera movement. Reviewers assign `pre_snap`, `snap`, and `ball_release` labels and fit the PTS map from reviewed correspondences, with separate held-out events. They do not produce a reviewed alignment or valid calibration.

### 3. Review in CVAT

Create a label schema with `player`, `official`, `football`, `ground_contact` points, and `field_landmark` points. Add player attributes for `team` (`DET`, `LAR`, or `unknown`), `visibility`, `jersey_readable`, `review_status`, `global_id`, and `cross_shot_review_status`. Add `identity_second_reviewer`, `identity_second_reviewed_at`, `identity_second_revision`, and `identity_second_confidence` for every reviewed cross-shot decision. Keep team and role as attributes rather than detector classes when the goal is to generalize to new teams.

For `ground_contact`, keep `tracklet_id` and `ground_contact_confidence`; for `field_landmark`, use the semantic `landmark_id` and `role` (`fit` or `withheld`). These attributes are declared in exported XML; configure them in the CVAT task schema before editing if CVAT asks for existing label definitions.

Review in this order:

1. Fix the first and last box of every track.
2. Correct rapid camera motion and non-linear movement with additional keyframes.
3. Split a track at an identity switch; merge only when the same player is supported by nearby evidence.
4. Mark a player `outside` when no visible box exists, rather than stretching a box across the gap.
5. Mark occlusion explicitly during blocking and tackles.
6. Inspect every shot boundary, wide formation, crossing, and pile at native resolution.
7. Review officials and sideline personnel as hard negatives so the detector learns the field region.

Interpolation is a labor-saving proposal. It is least trustworthy during a camera pan, contact, a player crossing another player, or a rapid scale change. Those intervals deserve dense keyframes.

### 4. Export two kinds of ground truth

Export reviewed boxes as COCO or YOLO for RF-DETR training. Export persistent tracks in CVAT native format or MOT-style form for HOTA, IDF1, fragmentation, and ID-switch evaluation. Keep detection labels and track labels separate: a frame can have a correct box even when its identity link is unknown.

For training data, sample diverse keyframes rather than every frame. For tracking evaluation, retain contiguous native-rate windows around snap, crossings, contact, and recovery from occlusion. Keep both views of a replayed play in one game/play split.

### 5. Use active learning to choose the next labels

After each local run, prioritize frames with:

- low RF-DETR confidence or large detector/tracker disagreement;
- new, lost, or rapidly changing tracklets;
- high box overlap between players;
- a camera cut, pan, zoom, or replay transition;
- a readable jersey number;
- a blocking pile, tackle, sideline exit, or distant wide-shot player.

FiftyOne can help curate these samples before sending them to CVAT. A practical first dataset is a few hundred reviewed frames plus several contiguous hard sequences from multiple games. The exact count matters less than viewpoint, lighting, uniform, and occlusion coverage.

### 6. Retrain and verify

Fine-tune RF-DETR on the reviewed COCO/YOLO export. Re-run the same clips with cached configurations, then compare the old and new detector on a held-out game. Do not tune against the test set. Apply the blog's horizontal flip, lighting, and motion-blur ideas only after checking that they preserve box geometry and do not make small jerseys unreadable.

The acceptance report should include visible-player precision and recall, HOTA, IDF1, fragmentation, ID switches, team-label accuracy and coverage, and failure examples. The [evaluation plan](evaluation-plan.md) contains proposed gates; the Roboflow tutorial's reported metrics are not substitutes for this project's held-out measurements.

## Repository components for this workflow

The versioned annotation contract is documented in [annotation-schema.md](annotation-schema.md).
Use `scripts/build_identity_review_pack.py` to extract exact, original-resolution frames,
source PTS, detector proposals, field-line intersections, and box-bottom contact hints.
Intersection semantics and contact confidence remain human decisions. Its output is an
unreviewed proposal pack; it cannot be passed to the evaluation loader until a human has
filled the shot, split, landmark, contact, and identity fields and marked the manifest
reviewed.

The current implementation includes these source-addressed components:

- `src/football_tracking/cvat.py`: source-hashed task-frame maps, CVAT video XML, and source-coordinate conversion;
- `scripts/export_cvat.py` and `scripts/import_cvat.py`: source-addressed proposal export and reviewed-manifest import;
- `scripts/propose_jersey_reads.py`: optional local OCR proposals tied to exact crops;
- `scripts/build_identity_review_queue.py`: source-addressed identity case ordering and contact sheets;
- `scripts/propose_timing_events.py`: bounded optical-flow motion-burst proposals for human event review.

Use the CVAT importer to retain corrected annotations as a source-addressed manifest; do not hand-edit `observations.csv` as a substitute for an annotation record. Keep reviewed corrections in a versioned overlay so raw inference can be reproduced and compared.
