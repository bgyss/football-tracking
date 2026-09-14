# McByte evaluation evidence

Status as of 2026-09-12: plumbing is implemented and unit-tested; real
integration and quality evaluation are not yet evidenced in this checkout.

## Implemented contract

- `--tracker mcbyte` is opt-in; BoT-SORT remains the parser default.
- The adapter converts each BGR source frame to one contiguous RGB array at the
  McByte boundary, advances empty frames, preserves source frame/PTS, and
  creates a fresh temporal backend for each shot.
- Mask-enabled mode explicitly sets `enable_mask_manager=True`, requires
  existing SAM/Cutie checkpoint paths, exposes its requested/effective mode in
  `metrics.json`, and fails rather than silently treating an OOM-degraded run
  as mask-active.
- The `mcbyte` dependency extra pins `trackers[mask]==2.6.0`; `evaluation`
  pins `trackeval==1.1.0`. The local installed wheel reported the expected
  `McByteTracker.update(detections, frame, timestamp)` and
  `McByteMaskConfig(device, sam_checkpoint_path, cutie_weights_path)` APIs.
- `--detection-cache` enables strict shared replay. It requires complete,
  unique native source-frame records (including explicit empties), matching
  source/detector/checkpoint/class/proxy provenance, and never regenerates a
  rejected cache.
- Reviewed MOT-style references use an explicit `reviewed: true` flag and
  preserve source-frame to one-based evaluator-frame conversion. Missing,
  unreviewed, ignored, and unlabeled data does not become negative ground
  truth. Reports include per-shot and aggregate DetA, AssA, HOTA, IDF1, ID
  switches, fragmentation, and coverage. When the `evaluation` extra is
  installed, the adapter translates the same frozen data into TrackEval 1.1.0
  sequence inputs and records its HOTA, Identity, and CLEAR outputs alongside
  the project coverage accounting.

## Commands run

```bash
UV_CACHE_DIR=.uv-cache uv lock
UV_CACHE_DIR=.uv-cache uv sync --extra dev --extra mcbyte --extra evaluation
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_tracking.py tests/test_detector.py tests/test_cli.py tests/test_evaluation.py -q
```

The full unit suite passed after the implementation, as did the top-level,
`run`, and `benchmark` CLI help checks plus `git diff --check`. Real-run
evidence must be recorded below before promotion claims.

The core OpenCV dependency is pinned as `opencv-python` rather than the
conflicting headless distribution because the optional Roboflow stack also
requires that distribution. Fresh `uv sync --extra dev` and combined
`uv sync --extra dev --extra mcbyte --extra evaluation` installs both retained
the `cv2` import and passed the full suite.

The CLI now applies a 2048 MiB default host-RSS guard through
`--max-memory-mb`, records peak/stage samples in run metrics, and rejects a
non-positive budget. It is an early-stop safeguard rather than a claim that
CUDA/MPS allocator requests cannot fail before control returns to Python.

An actual local `trackers==2.6.0` mask-free McByte smoke test also initialized
the pinned backend, processed an observed frame plus an explicit empty frame,
and reset into a new shot namespace at `60000/1001` fps. Its effective mode
reported `masks_requested: false` and `masks_active: false`. This verifies only
the mask-free adapter/API path; it is explicitly not a mask-assisted result.

## Evidence gates still open

| State | Current status | Required input/evidence |
| --- | --- | --- |
| Plumbing ready | Verified locally | Full prescribed suite, CLI help, and diff check completed; see commands above |
| Real integration verified | Not evaluated | Authorized source video, frozen real RF-DETR cache, SAM vit_b and Cutie base-mega checkpoints with source/hash/license records, supported device, and mask-active run output |
| Quality evaluated | Not evaluated | Human-reviewed references, frozen evaluator configuration, all comparison variants, metrics, plots, and failure clips |
| Promotion candidate | Not evaluated | Per-shot comparative gates plus the broader game-disjoint held-out experiment |

No McByte quality or mask-active runtime claim is made here. The originating
checkout contains the sample video with SHA-256
`3e7e9ada26b8adc94c98094db42130767c4871c54104c9ea49874d65393fa374` and
several 1,424-frame legacy caches, but those caches predate strict provenance:
their metadata lacks checkpoint-content hash, class mapping, and proxy/real
declaration, so this implementation correctly rejects them for controlled
comparison. The mask checkpoints are now present and verified under ignored
`artifacts/mcbyte-evaluation/weights/`; no reviewed identity reference was
found.

Two earlier local full RF-DETR cache rebuilds used the authorized source and
the existing generic checkpoint (whose upstream MD5 validation reported
success), but the local runner ended both processes before output artifacts
were written and provided no application-level exception. They are therefore
not integration evidence. The required resume input remains a completed
strict-provenance cache plus mask weights and reviewed reference data.

The resumable rebuild subsequently completed all 1,424 native frames. The
merged cache is `artifacts/mcbyte-evaluation/strict-real-cache.jsonl` (1,425
JSONL lines including metadata), with source hash
`3e7e9ada26b8adc94c98094db42130767c4871c54104c9ea49874d65393fa374`, RF-DETR
checkpoint SHA-256
`d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33`, class
mapping `{"1":"person"}`, and `proxy: false`.

The safe regenerated artifact is
`artifacts/mcbyte-evaluation/mcbyte-mask-off/annotated.mp4`. FFprobe reports
1,424 video frames, 23.757090 seconds, 59.94 fps, and an audio stream; source
video reports 1,424 frames and 23.757067 seconds. This is an actual pinned
McByte run with masks explicitly off, not evidence for mask assistance.

Mask-active construction was attempted with the verified SAM/Cutie weights,
but macOS could not enforce the 1.6 GiB pre-allocation ceiling and the process
reached approximately 5.6 GiB RSS. The safety gate now refuses mask-enabled
construction on that host unless `--allow-unbounded-memory` is explicitly
provided; that guarded attempt emitted no result and made no quality claim.

That memory-limited attempt was later rerun explicitly with
`--allow-unbounded-memory` at the user's direction. The completed artifact is
`artifacts/mcbyte-evaluation/mcbyte-mask-on-unbounded/annotated.mp4`; its
manifest reports 1,424 source frames, two shot-local McByte states, and
`masks_requested: true`, `masks_active: true`, `consecutive_mask_failures: 0`
for both shots. Tracking took 3,771.19 seconds and export 10.42 seconds. The
recorded process peak was 20,125.19 MiB, so this is not a safe default for the
current Mac. The run's quality status is `not_evaluated` because no reviewed
identity reference is available; it must not be interpreted as a football
accuracy result.
