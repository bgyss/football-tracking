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

The run writes `annotated.mp4`, `observations.csv`, `observations.parquet`, `identities.json`, `field-view.png`, `trajectories.csv`, `shots.json`, `calibration.json`, `identity-links.json`, `review.json`, `metrics.json`, `run-manifest.json`, and the detector cache. `trajectories.csv` always contains image-space contact points; its yard columns are populated only with a valid shot-specific landmark file passed through `--calibration`. Image-space trails are always available in the annotated video.

## Local evidence from the supplied clip

The implementation was exercised on the full 1,424-frame clip with the generic RF-DETR Small checkpoint, CPU BoT-SORT, and the manually verified frame-712 shot boundary. It produced 17,222 observations across 59 shot-local tracklets. RF-DETR inference took 131.43 seconds, BoT-SORT association 8.00 seconds, and export 9.15 seconds on the Apple Silicon development machine. The annotated output retained 1,424 frames, 23.757 seconds, and the source audio.

The same cached detections produced 18,221 observations and 50 tracklets with ByteTrack, and 24,923 observations and 160 tracklets with the deterministic IoU fallback. These are association and throughput comparisons without tracking ground truth; the [evaluation plan](docs/evaluation-plan.md) defines the annotations and HOTA/IDF1 gates needed for accuracy claims. The generic checkpoint is not football fine-tuned, so its count and identity output are integration evidence only.

The previous design-only status is superseded by the implementation and benchmark artifacts in this repository. No GPT-6 Astra requests were made; the optional semantic layer remains disabled by default.
