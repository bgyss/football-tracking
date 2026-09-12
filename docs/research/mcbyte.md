# McByte integration research

Checked 2026-09-12. This is source inspection and upstream evidence, not a local tracking result. The proposed evaluation is in [the evaluation plan](../mcbyte-evaluation-plan.md).

## Version to target

The project already pins `trackers==2.6.0`. That release includes public exports `McByteTracker` and `McByteMaskConfig`; no development-branch dependency is necessary. The GitHub `2.6.0` tag resolved to `0e839f348d8bf4ed09eea9f3bef58fd5f95dca3f`, and its release is dated 2026-08-06. The linked `develop` documentation is moving material (`develop` resolved to `80a09d683642bb0a43521286709e6c6da14c00d0` during this check). Verify the installed wheel's version and signatures before implementation: this research inspected the release-tag source, not an installed wheel. [Release](https://github.com/roboflow/trackers/releases/tag/2.6.0), [public exports](https://github.com/roboflow/trackers/blob/0e839f348d8bf4ed09eea9f3bef58fd5f95dca3f/src/trackers/__init__.py).

The requested rendered documentation URL did not load in the research browser; its source was retrieved from the official repository instead. Release-tag source takes precedence where examples and tables disagree with implementation.

## Verified API and critical behavior

The public construction and update shape is:

```python
from trackers import McByteMaskConfig, McByteTracker

tracker = McByteTracker(
    frame_rate=actual_source_fps,
    enable_mask_manager=True,
    mask_config=McByteMaskConfig(
        device=explicit_device,
        sam_checkpoint_path=sam_checkpoint,
        cutie_weights_path=cutie_checkpoint,
    ),
)
tracked = tracker.update(detections, frame=frame_rgb)
tracker.reset()
```

This is an illustrative API contract, not a runnable local command. `detections` is `supervision.Detections`; outputs are new detections carrying `tracker_id`, including `-1` for unconfirmed/unmatched detections. **McByte requires RGB frames**, unlike the base tracker's BGR convention. The documentation video example passes BGR and must not be copied literally. Missing frames skip both masks and camera-motion compensation. Masks are **disabled by default**, so `McByteTracker()` alone does not exercise the principal mask-conditioned hypothesis. [Pinned tracker source](https://github.com/roboflow/trackers/blob/0e839f348d8bf4ed09eea9f3bef58fd5f95dca3f/src/trackers/core/mcbyte/tracker.py).

Pinned defaults include `enable_cmc=True`, `cmc_method="sparseOptFlow"`, `minimum_iou_threshold_first_assoc=0.1`, and `minimum_mask_creation_frames=3`. `McByteMaskConfig` defaults to `device="auto"` (CUDA if available, otherwise CPU; MPS never selected automatically), `cutie_use_amp=False`, `cutie_max_internal_size=480`, `cutie_mem_every=10`, and `cutie_use_long_term=True`. These differ from the documentation's older default table. Record resolved values explicitly in run manifests. The implementation supports `timestamp`, but warns that mask propagation advances once per update while Kalman state scales with elapsed time; sparse or irregular sampling can desynchronize them. [Pinned tracker source](https://github.com/roboflow/trackers/blob/0e839f348d8bf4ed09eea9f3bef58fd5f95dca3f/src/trackers/core/mcbyte/tracker.py).

Out-of-memory errors cause per-frame IoU-only fallback; three consecutive failures disable mask association for the remainder of the run. `reset()` resets lifecycle and mask state and can restore the disabled manager. A successful process exit therefore does not establish a full-mask run: capture failures, mask availability and actual mask usage, and invalidate or separately label degraded runs. [Fallback implementation](https://github.com/roboflow/trackers/blob/0e839f348d8bf4ed09eea9f3bef58fd5f95dca3f/src/trackers/core/mcbyte/tracker.py).

## Dependencies, weights and hardware

The release defines a `mask` extra: `trackers[mask]==2.6.0` adds `torch`, `torchvision`, `rf-segment-anything>=1.0`, and `rf-cutie[inference]>=1.0.0`. Prefer this released packaging over older PR instructions to clone and edit Cutie's dependencies. Resolve exact versions into the project's lockfile during integration and validate compatibility with existing RF-DETR dependencies. [Pinned packaging](https://github.com/roboflow/trackers/blob/0e839f348d8bf4ed09eea9f3bef58fd5f95dca3f/pyproject.toml).

The built-in SAM asset is `vit_b` / `sam_vit_b_01ec64.pth`; Cutie's is `base-mega` / `cutie-base-mega.pth`. Default locations are `models/sam/` and `models/cutie/`, relative to the working directory. Construction may download weights. A supplied Cutie file is honored and missing files fail, but the SAM implementation still calls its download/checksum helper even with an explicit path, validating against the known default asset. Consequently, merely supplying paths is not a sufficient no-download guarantee: preflight existence and expected checksums before construction, and keep acquisition explicit. Record SHA-256 and source URLs in the experiment manifest. [SAM asset handling](https://github.com/roboflow/trackers/blob/0e839f348d8bf4ed09eea9f3bef58fd5f95dca3f/src/trackers/core/mcbyte/masks/sam.py), [Cutie asset handling](https://github.com/roboflow/trackers/blob/0e839f348d8bf4ed09eea9f3bef58fd5f95dca3f/src/trackers/core/mcbyte/masks/cutie.py).

CUDA is the practical first performance target if an authorized host is available. The author reports CPU execution as very slow; source comments report MPS slower than CPU for this pipeline, but those are upstream observations, not measurements on this project's host. No trustworthy minimum VRAM or football-specific throughput was established. Measure initialization, warmup, tracker-only and end-to-end time, and peak memory. Cutie cost grows with active object count and frame resolution; changes in internal resolution, AMP or memory settings must be separately identified accuracy/performance variants. [Author's integration report](https://github.com/roboflow/trackers/pull/513), [pinned configuration](https://github.com/roboflow/trackers/blob/0e839f348d8bf4ed09eea9f3bef58fd5f95dca3f/src/trackers/core/mcbyte/tracker.py).

Trackers declares Apache-2.0; SAM's upstream README explicitly licenses its model under Apache-2.0; Cutie's repository and `rf-cutie` package declare MIT. These are provenance findings, not a blanket conclusion about every transitive dependency or checkpoint redistribution. Record the exact installed distributions, checkpoint terms and attribution before distribution. No datasets are needed just to instantiate the tracker. [Trackers license declaration](https://github.com/roboflow/trackers/blob/0e839f348d8bf4ed09eea9f3bef58fd5f95dca3f/pyproject.toml), [SAM license](https://github.com/facebookresearch/segment-anything#license), [Cutie license](https://github.com/hkchengrex/Cutie/blob/main/LICENSE), [rf-cutie metadata](https://pypi.org/project/rf-cutie/).

## Evidence for the identity-stability hypothesis

Roboflow's McByte page attributes the following **author-reported** comparisons to PR #513. Both use default parameters; the baseline is BoT-SORT without re-identification and McByte uses masks. They motivate a local experiment but do not measure this project's footage, detector, settings, or current pinned runtime defaults. [Benchmark report](https://github.com/roboflow/trackers/pull/513).

| Dataset | BoT-SORT HOTA → McByte | BoT-SORT IDF1 → McByte |
| --- | --- | --- |
| SportsMOT | 73.8 → 76.5 | 73.4 → 76.9 |
| SoccerNet-tracking | 84.5 → 85.0 | 79.3 → 79.9 |
| DanceTrack | 57.8 → 67.2 | 57.9 → 68.6 |
| MOT17 | 63.7 → 64.1 | 78.7 → 79.7 |

These tables do not report an identity-switch count. Better IDF1 does not by itself establish a particular reduction in switches. The broader comparison page uses v2.3.0 and YOLOX detections or SoccerNet oracle boxes, reinforcing the need to avoid transferring scores to RF-DETR American-football video. [Comparison methodology](https://github.com/roboflow/trackers/blob/0e839f348d8bf4ed09eea9f3bef58fd5f95dca3f/docs/trackers/comparison.md).

The original paper studies sports and pedestrian datasets and describes propagated masks as association cues without additional training. That is not proof of stable player identities through this project's pileups, hard cuts or shot boundaries. McByte++ is a separate later research system with re-identification; its claims should not be attributed to Roboflow's `McByteTracker`. [Original paper](https://arxiv.org/abs/2506.01373), [McByte++ paper](https://arxiv.org/abs/2608.15688).

The project experiment should compare current BoT-SORT, mask-free McByte, and full-mask McByte on identical detector outputs and dense decoded frames, with independent shot-local state. Measure ID switches and fragmentation against human-reviewed identities as well as IDF1/HOTA, detection coverage and compute cost. Synthetic fixtures prove interface behavior; a mask-free run or unannotated overlay cannot establish the proposed accuracy improvement.
