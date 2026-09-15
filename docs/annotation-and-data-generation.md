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

CVAT Community is MIT-licensed, self-hostable, and keeps the video in the local annotation environment. Its automatic-annotation interface can also run a custom tracking function under your control. A future adapter could implement CVAT's tracking-function protocol with `spec`, `init_tracking_state`, and `track`; importing reviewed preannotations is simpler for the first iteration. [CVAT auto-annotation API](https://docs.cvat.ai/docs/api_sdk/sdk/auto-annotation/)

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

The run produces `detections.jsonl`, `observations.csv`, `observations.parquet`, `trajectories.csv`, and `annotated.mp4`. The future CVAT adapter should read `observations.csv` and create one `source="auto"` CVAT track per `tracklet_id`. Use `tracklet_id` as the proposal identity because the current `player_id` is deterministic per tracklet and cross-view replay identity is intentionally unresolved.

The converter should preserve original frame numbers and boxes, attach the detector score as an attribute, and mark a box as a keyframe when a track starts, ends, changes sharply, or crosses an occlusion boundary. It should never silently discard detector candidates or rewrite the raw cache.

### 3. Review in CVAT

Create a label schema with `player`, `official`, and `football`. Add attributes for `team` (`DET`, `LAR`, or `unknown`), `occluded`, `visibility`, `jersey_readable`, and `review_status`. Keep team and role as attributes rather than detector classes when the goal is to generalize to new teams.

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

## Repository changes that make this workflow easy

The versioned annotation contract is documented in [annotation-schema.md](annotation-schema.md).
Use `scripts/build_identity_review_pack.py` to extract exact, original-resolution frames
and detector proposals. Its output is an unreviewed proposal pack; it cannot be passed to
the evaluation loader until a human has filled the shot, split, landmark, contact, and
identity fields and marked the manifest reviewed.

The current code already has a clean seam for an annotation adapter. The next small additions should be:

- `src/football_tracking/cvat.py`: serialize and parse reviewed CVAT tracks;
- `scripts/export_cvat.py`: convert a run's `observations.csv` into CVAT preannotations;
- `scripts/import_cvat.py`: convert corrected tracks into COCO/YOLO and MOT-style ground truth;
- `docs/annotation-schema.md`: freeze labels, attributes, visibility rules, and split policy;
- an active-learning report that emits the frame list above from detector and tracker uncertainty.

Until those adapters exist, do not hand-edit `observations.csv` as a substitute for an annotation record. Keep reviewed corrections in a versioned overlay so raw inference can be reproduced and compared.
