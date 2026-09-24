# Local All-22 staging run: real detections, review proposals, open identity gate

Run date: 2026-09-23. This is evidence from the local Lions–Rams sample, not a
reviewed accuracy benchmark. Generated files are in ignored
`artifacts/all22-real-2026-09-23/` and `artifacts/all22-stage-2026-09-23/`.
The video and checkpoint are local files; the run manifest records zero API
requests. No CVAT server was started, and no hosted identity service was used.

## Input and inference

The 1280×720 sample has 1,424 frames at 60000/1001 FPS and SHA-256
`3e7e9ada26b8adc94c98094db42130767c4871c54104c9ea49874d65393fa374`.
The known camera cut is source frame 712. The run used RF-DETR Small with an
existing generic checkpoint (SHA-256
`d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33`),
threshold 0.1, and BoT-SORT with camera-motion compensation. It was a fresh
inference pass, not a cache replay. The detector is **not football fine-tuned**.

| Observed artifact | Result |
| --- | --- |
| `metrics.json` | 18,023 observations, 57 shot-local tracklets, zero duplicate track IDs within a frame; 27 tracklets in shot 0 and 30 in shot 1 |
| `run-manifest.json` | `complete_with_unresolved`; 190.4 s decode and track, including 160.3 s detection; 1,100.5 MiB sampled peak RSS |
| `artifact-validation.json` | `valid` structural artifact set |
| `calibration-quality.json` | `not_provided`; no observation has field-yard coordinates |
| `identity-links.json` | `not_attempted`; 57 player IDs for 57 tracklets, zero spanning the cut |
| `tracking-evaluation.json` | `not_evaluated`; no reviewed reference or cross-shot score |

An observation count and a clean artifact set do not establish player accuracy
or track purity. The second view replays the first play from the end zone; its
tracklets cannot be joined by continuing image-space trajectories across frame
712. The [cross-shot evaluation contract](../evaluation-plan.md) still applies.

## Automated local staging

`scripts/stage_local_review.py` reused the validated run and produced:

- Two source-addressed CVAT proposal bundles and frame maps, each covering 712
  task frames. Shot 0 maps task frame 0 to source frame/PTS `0/0`; shot 1 maps
  task frame 0 to source frame/PTS `712/712712`. Both maps retain the source
  hash and integer PTS. Their boxes and landmark points are unreviewed.
- One two-shot play-window candidate, eight extracted source frames, and 255
  unreviewed field-intersection hints across those frames. These hints have no
  assigned yard-line semantics or independent withheld-landmark validation.
- Sixteen motion-burst candidates and three camera-motion keyframe candidates.
  They are not classified snap or ball-release events.
- 296 local Tesseract crops over 57 tracklets; 93 crops returned at least one
  candidate digit. Single digits recur on many unrelated tracklets. Only two
  two-digit numbers repeated within any one tracklet, both in shot 1; there
  was no repeat-supported two-digit cross-shot match. OCR was kept as
  `reviewed: false` and did not assign identity.
- An empty identity review queue because the resolver had no reviewed play-time
  alignment or shot-specific validated calibration from which to form field
  candidates. `readiness.json` is `not_ready`: reviewed annotations, calibration,
  timing, and reference are missing.

The stage output is useful for targeted review, but it does not demonstrate
cross-shot identity on this clip. The next minimal evidence slice is to label
semantic field points plus an independent withheld point at relevant keyframes
in **both** shots, confirm corresponding play-time events, and independently
review shared-player IDs. Then rerun with the resulting calibration timeline,
play alignment, and reference. A local CVAT import/edit/export round trip and
false-merge measurement remain open gates. The [Astra comparison note](astra-identity-options-2026-09-23.md)
keeps optional API experiments separate from this local run.

## Reproduction and verification

From a checkout with the sample in `data/` and an installed local checkpoint:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/stage_local_review.py \
  --source data/all-22-lions-rams-sample.mp4 \
  --run-dir artifacts/all22-run \
  --output artifacts/all22-review-stage \
  --manual-cut 712 \
  --detector-checkpoint /path/to/local-rf-detr-small.pth
```

The command runs inference if `--run-dir` lacks a manifest, or validates and
reuses that run. The focused stage tests and full repository suite passed with
`PYTHONPATH=src:. UV_CACHE_DIR=.uv-cache uv run --no-sync pytest -q` in the
installed development environment. The real stage command exited 0 and wrote
`stage-summary.json` with `proposals_ready_review_required` and
`readiness_status: not_ready`. This verifies the local proposal path; it does
not substitute for reviewed source labels or a live CVAT round trip.
The one-command inference path was also exercised on a four-frame local video
with the synthetic proxy detector; that smoke run checks orchestration only
and is not part of the real-footage measurements above.
