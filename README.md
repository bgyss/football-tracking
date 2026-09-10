# American football player tracking

Design for offline All-22 analysis using Roboflow RF-DETR, BoT-SORT, supervision, and selective GPT-6 Astra identity assistance.

The first version prioritizes **stable anonymous player IDs, teams, and trajectories**, as requested. Named-player recognition and ball possession are later extensions.

- [System design](docs/system-design.md): architecture, component choices, data contracts, runtime strategy, and milestones.
- [Evaluation plan](docs/evaluation-plan.md): measurable acceptance criteria and fair comparisons.
- [Tool research](docs/research/tracking-tools.md): primary sources, current APIs, alternatives, and licenses.
- [Sample footage assessment](docs/evidence/footage-assessment.md): verified metadata and inspected frames.
- [Accuracy, calibration, and replay guide](docs/accuracy-calibration-replay.md): how to move from the baseline to football accuracy, yard-space calibration, and cross-view identity.
- [Local annotation and data generation](docs/annotation-and-data-generation.md): CVAT/SAM2/FiftyOne workflow for reviewing initial tracks and fine-tuning RF-DETR without hosted Roboflow services.

Status: **implemented and locally benchmarked**. The supplied [sample video](data/all-22-lions-rams-sample.mp4) is 23.76 seconds and contains a sideline view followed by an apparent end-zone replay. The implementation includes local model inference and proxy benchmarks; no GPT-6 Astra request or footage upload was performed.

## Run locally

Create the isolated environment and install the local and Roboflow adapters:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev --extra roboflow
```

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

The run writes `annotated.mp4`, `observations.csv`, `observations.parquet`, `identities.json`, `field-view.png`, `trajectories.csv`, `shots.json`, `calibration.json`, `review.json`, `metrics.json`, `run-manifest.json`, and the detector cache. `trajectories.csv` always contains image-space contact points; its yard columns are populated only with a valid shot-specific landmark file passed through `--calibration`. Image-space trails are always available in the annotated video.

## Local evidence from the supplied clip

The implementation was exercised on the full 1,424-frame clip with the generic RF-DETR Small checkpoint, CPU BoT-SORT, and the manually verified frame-712 shot boundary. It produced 17,222 observations across 59 shot-local tracklets. RF-DETR inference took 131.43 seconds, BoT-SORT association 8.00 seconds, and export 9.15 seconds on the Apple Silicon development machine. The annotated output retained 1,424 frames, 23.757 seconds, and the source audio.

The same cached detections produced 18,221 observations and 50 tracklets with ByteTrack, and 24,923 observations and 160 tracklets with the deterministic IoU fallback. These are association and throughput comparisons without tracking ground truth; the [evaluation plan](docs/evaluation-plan.md) defines the annotations and HOTA/IDF1 gates needed for accuracy claims. The generic checkpoint is not football fine-tuned, so its count and identity output are integration evidence only.

The previous design-only status is superseded by the implementation and benchmark artifacts in this repository. No GPT-6 Astra requests were made; the optional semantic layer remains disabled by default.
