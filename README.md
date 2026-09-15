# American football player tracking

Design for offline All-22 analysis using Roboflow RF-DETR, BoT-SORT, supervision, and selective GPT-6 Astra identity assistance.

The first version prioritizes **stable anonymous player IDs, teams, and trajectories**, as requested. Named-player recognition and ball possession are later extensions.

- [System design](docs/system-design.md): architecture, component choices, data contracts, runtime strategy, and milestones.
- [Evaluation plan](docs/evaluation-plan.md): measurable acceptance criteria and fair comparisons.
- [Tool research](docs/research/tracking-tools.md): primary sources, current APIs, alternatives, and licenses.
- [McByte evaluation plan](docs/mcbyte-evaluation-plan.md): controlled identity-switch experiment and [integration goal prompt](docs/goals/mcbyte-integration.md) for a new task.
- [Sample footage assessment](docs/evidence/footage-assessment.md): verified metadata and inspected frames.
- [Accuracy, calibration, and replay guide](docs/accuracy-calibration-replay.md): how to move from the baseline to football accuracy, yard-space calibration, and cross-view identity.
- [Local annotation and data generation](docs/annotation-and-data-generation.md): CVAT/SAM2/FiftyOne workflow for reviewing initial tracks and fine-tuning RF-DETR without hosted Roboflow services.
- [Annotation schema](docs/annotation-schema.md): source-hashed reviewed boxes, NFL field landmarks, timing, and identity labels.

Status: **implemented and locally benchmarked**. The supplied [sample video](data/all-22-lions-rams-sample.mp4) is 23.76 seconds and contains a sideline view followed by an apparent end-zone replay. The implementation includes local model inference and proxy benchmarks; no GPT-6 Astra request or footage upload was performed.

## Run locally

Create the isolated environment and install the local and Roboflow adapters:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev --extra roboflow
```

McByte is opt-in and never changes the BoT-SORT default. Its full mask-assisted
mode has separate local dependencies and requires already-downloaded SAM and
Cutie weights; the command performs no intentional weight acquisition:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev --extra roboflow --extra mcbyte --extra evaluation
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import inspect
import trackers
print(inspect.signature(trackers.McByteTracker))
print(inspect.signature(trackers.McByteMaskConfig))
PY
```

For a controlled real-cache replay, first create and preserve a complete cache
from a real detector run, then provide it explicitly. The shared cache is
rejected if its source hash, checkpoint hash, detector settings, player class
mapping, proxy provenance, or native-frame coverage differs.

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run \
  --input data/all-22-lions-rams-sample.mp4 \
  --output artifacts/mcbyte-evaluation/mcbyte-mask-on \
  --detection-cache artifacts/frozen-real-detections.jsonl \
  --detector rfdetr --detector-checkpoint /path/to/football-rfdetr.pth \
  --detector-class-mapping /path/to/detector-class-mapping.json \
  --tracker mcbyte --mcbyte-masks on --mcbyte-device cpu \
  --mcbyte-sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --mcbyte-cutie-checkpoint /path/to/cutie-base-mega.pth \
  --reviewed-reference /path/to/reviewed-mot-reference.json \
  --manual-cut 712
```

`--mcbyte-masks off` records the supported McByte ablation. A run with masks
disabled after runtime fallback is labeled degraded and cannot satisfy the
mask-active gate. Without `--reviewed-reference`, `tracking-evaluation.json`
states `not_evaluated`; no overlay or track count is treated as identity truth.
The class-mapping file is a non-empty JSON object, for example
`{"1": "person"}` for the generic checkpoint, and is compared exactly with
the cache provenance before replay.

## Cross-shot replay identity

Without `--play-alignment`, every run — including the default RF-DETR/BoT-SORT
path — assigns each shot-local tracklet its own anonymous `player_id` and
writes `identity-links.json` with `status: not_attempted`. No tracklets are
joined across the frame-712 cut. To attempt a constrained cross-shot join,
supply a reviewed play-time alignment, shot-specific calibration, and a
reviewed reference:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run \
  --input data/all-22-lions-rams-sample.mp4 \
  --output artifacts/cross-shot \
  --detector rfdetr --detector-checkpoint /path/to/football-rfdetr.pth \
  --tracker botsort --manual-cut 712 \
  --calibration /path/to/shot-landmarks.json \
  --play-alignment /path/to/reviewed-snap-anchors.json \
  --reviewed-reference /path/to/reviewed-mot-reference.json
```

Without `--calibration` supplying valid field positions for both aligned
shots, the resolver abstains by design and `identity-links.json` records
`status: abstained` with no merges. `tracking-evaluation.json` reports
`cross_shot.status: not_evaluated` unless the reviewed reference carries a
`cross_shot_identity` map. See
[docs/evidence/cross-shot-identity.md](docs/evidence/cross-shot-identity.md)
for the measured state of this milestone on the current assets.

When a reviewed reference is supplied, `tracking-evaluation.json` also contains a
`promotion_gate`. It stays `not_evaluated` until standard TrackEval metrics, valid
calibration, and cross-shot precision/coverage evidence are all present; proxy or
missing-reference results never count as acceptance.
The same promotion decision is copied into `review.json` for quick inspection.

For a PTS-based alignment to be eligible for identity, include at least two
reviewed correspondences per shot plus separate `validation_correspondences`;
the latter must pass the configured timing residual gate. A snap-only anchor
remains an explicitly unvalidated equal-rate compatibility mode.

An alignment file may contain a reviewed `plays` array. Select one play per
bounded run with `--play-id`; omitting it from a multi-play file fails rather
than silently choosing the first play.

For a reviewed alignment set with `shot_ranges`, `batch` runs each play in a separate
bounded directory and writes a top-level `batch.json` summary:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking batch \
  --input data/all-22-lions-rams.mp4 \
  --output artifacts/game-batch \
  --detector synthetic --tracker iou \
  --play-alignment /path/to/reviewed-play-alignments.json
```

`batch.json` distinguishes execution completion from identity completion and uses
`complete_with_unresolved` until every child has a resolved identity result.

For a tracker that fragments within one view, prepare a reviewed split overlay
and pass it with `--reviewed-splits`. The raw tracker observations and cache are
preserved; only the derived identity segments use the reviewed split IDs:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run \
  --input data/all-22-lions-rams-sample.mp4 \
  --output artifacts/cross-shot-refined \
  --detector synthetic --tracker iou --manual-cut 712 \
  --reviewed-splits /path/to/reviewed-splits.json
```

The split file must contain `{"reviewed": true, "source_sha256": "...", "splits": ...}`
matching the input video, and every observed source tracklet frame must belong to exactly
one reviewed segment.

For moving cameras, `--calibration` also accepts a schema-v2 timeline whose
keyframes carry `pts_start`, `pts_end`, fitting landmarks, and independent
`withheld_image_points`/`withheld_field_points`. Only keyframes that pass the
withheld geometry gate are eligible for cross-shot identity; gaps remain
unresolved. A generated timeline also carries the input `source_sha256` so it
cannot be applied accidentally to another video.

To start reviewing the supplied long recording, extract exact scouting frames into an
ignored pack (this does not upload or modify the source video):

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/build_identity_review_pack.py \
  --input data/all-22-lions-rams.mp4 \
  --output artifacts/full-game-calibration-review-pack --exact-pts \
  --frame 1798 --frame 2098 --frame 2398
```

The pack is marked `reviewed: false` until a human records shot intervals, semantic field
landmarks, timing correspondences, and identity labels.

Generate a durable, unreviewed shot inventory before selecting paired plays:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/build_play_inventory.py \
  --input data/all-22-lions-rams.mp4 \
  --output artifacts/full-game-shot-inventory.json
```

The inventory records candidate boundaries, source frame/PTS ranges, and confidence. A
reviewer must confirm cuts, camera labels, play IDs, and split assignments before using it.

Create a strict annotation-manifest template from that pack by declaring the reviewed shot
ranges and camera labels:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/build_annotation_manifest.py \
  --review-pack artifacts/full-game-calibration-review-pack/review-pack.json \
  --output artifacts/full-game-calibration-review-pack/manifest-template.json \
  --shot shot-0:1798:2398:sideline:play-0042:development
```

The generated template remains `reviewed: false` until boxes, landmarks, contacts, timing,
and identity labels have been reviewed.

After review, fit the timeline directly from the manifest:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/fit_calibration_timeline.py \
  --annotations /path/to/reviewed-manifest.json \
  --source data/all-22-lions-rams.mp4 \
  --output artifacts/calibration-timeline.json
```

Long recordings should be processed in source-frame windows with `--start-frame` and
`--end-frame`. Window outputs record their scope in `metrics.json`; a partial detector
cache is never treated as a complete cache for another window, and windowed annotated
videos intentionally omit source audio until an offset-aware muxer is added. On this
constant-rate All-22 source, bounded windows derive PTS from the declared frame step;
full-source runs retain the exact ffprobe PTS index.

## Memory guard

Every run has a 2048 MiB host-process peak-RSS budget by default. It samples
after backend initialization, during native-frame tracking, and after export;
crossing the budget stops the run with an explicit error and records the peak
and stage samples in `metrics.json` for completed runs. Use a lower explicit
budget when working on a constrained machine:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run \
  --input data/all-22-lions-rams-sample.mp4 \
  --output artifacts/memory-guarded \
  --detector synthetic --tracker iou --max-memory-mb 1024
```

This guards host RSS, not an allocator request that may fail inside CUDA or
MPS before Python regains control. Keep McByte on its default explicit `cpu`
device unless accelerator capacity has been checked, and use a lower source
resolution or a smaller mask configuration when a full mask run exceeds the
budget.

Run the real RF-DETR/BoT-SORT path with a local checkpoint. The command below uses the generic Roboflow pretrained RF-DETR Small weights as an integration smoke test; a football-fine-tuned checkpoint should be supplied for production tracking:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run \
  --input data/all-22-lions-rams-sample.mp4 \
  --output artifacts/football-botsort \
  --detector rfdetr \
  --detector-checkpoint /path/to/football-rfdetr.pth \
  --model-size small \
  --tracker botsort \
  --device cpu \
  --manual-cut 712
```

For a repeatable plumbing run without model weights, use `--detector synthetic --tracker iou`; its outputs are explicitly marked as proxy results. The benchmark command compares ByteTrack, BoT-SORT, and the fallback tracker on the same proxy detections and records unavailable model gates instead of inventing scores:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking benchmark \
  --input data/all-22-lions-rams-sample.mp4 \
  --output artifacts/benchmark \
  --detector synthetic \
  --tracker botsort \
  --manual-cut 712
```

The run writes `annotated.mp4`, `observations.csv`, `observations.parquet`, `identities.json`, `field-view.png`, `trajectories.csv`, `play-trajectories.csv`, `shots.json`, `calibration.json`, `calibration-quality.json`, `tracklet-refinement.json`, `identity-links.json`, `analysis-config.json`, `review.json`, `metrics.json`, `artifact-validation.json`, `run-manifest.json`, and the detector cache. `analysis-config.json` is the frozen, schema-versioned provenance record for reproducing the analysis; `artifact-validation.json` checks cross-file consistency. `trajectories.csv` always contains image-space contact points; its yard columns are populated only with a valid shot-specific landmark file passed through `--calibration`. `play-trajectories.csv` is the deduplicated, aligned play-time view and preserves the source shot/frame selected for each bin. Image-space trails are always available in the annotated video.

## Local evidence from the supplied clip

The implementation was exercised on the full 1,424-frame clip with the generic RF-DETR Small checkpoint, CPU BoT-SORT, and the manually verified frame-712 shot boundary. It produced 17,222 observations across 59 shot-local tracklets. RF-DETR inference took 131.43 seconds, BoT-SORT association 8.00 seconds, and export 9.15 seconds on the Apple Silicon development machine. The annotated output retained 1,424 frames, 23.757 seconds, and the source audio.

The same cached detections produced 18,221 observations and 50 tracklets with ByteTrack, and 24,923 observations and 160 tracklets with the deterministic IoU fallback. These are association and throughput comparisons without tracking ground truth; the [evaluation plan](docs/evaluation-plan.md) defines the annotations and HOTA/IDF1 gates needed for accuracy claims. The generic checkpoint is not football fine-tuned, so its count and identity output are integration evidence only.

The previous design-only status is superseded by the implementation and benchmark artifacts in this repository. No GPT-6 Astra requests were made; the optional semantic layer remains disabled by default.
