# American football tracking system design

Design date: September 9, 2026. Status: proposed architecture, ready for implementation review.

## Recommendation

Build a local, offline, two-pass pipeline: **RF-DETR detects players; Roboflow's BoT-SORT implementation builds short tracks; a football-specific identity resolver joins compatible tracks; supervision renders results.** Add GPT-6 Astra as a selective source of identity evidence when local matching is ambiguous. Calibrate the field separately so trajectories describe player motion in yards despite camera movement.

The first deliverable is an annotated version of `data/all-22-lions-rams-sample.mp4`, a field-view trajectory visualization, machine-readable observations and identities, and an evaluation report. Prioritize stable anonymous IDs such as `DET-P07`, team membership, and trajectories. Jersey numbers can help identity matching without requiring named-player recognition.

This preserves the division of labor in the user-supplied basketball example while adding football-specific handling for camera cuts, replayed plays, crowded blocking, off-screen players, and field geometry. The quoted $0.20 and 16 seconds per video second are user-provided observations about that demonstration, not validated performance for this design. The linked X page could not be retrieved directly.

## What the sample changes

The [footage assessment](evidence/footage-assessment.md) establishes a 23.757-second, 1280 × 720, 59.94 fps clip, with a visually verified cut at 11.878533 seconds. The first shot is a wide sideline view; the second appears to replay the play from the end zone. Zoom changes, referees, sideline people, small jerseys, camera cables, and severe player overlap are visible.

Consequences: track each shot independently, keep media time separate from play time, and only join cross-view identities with evidence. Never append the replay to the original trajectory as though it were a continuation. The end-zone view does not show every participant, so do not manufacture 22 boxes or trajectories.

## Approach comparison

| Approach | Advantages | Limitations | Decision |
| --- | --- | --- | --- |
| RF-DETR + Roboflow BoT-SORT + separate identity resolver | Local detection, camera-motion compensation, inspectable identity decisions | Requires football data and explicit re-identification logic | Recommended |
| RF-DETR + ByteTrack | Simpler control experiment; official Roboflow American-football example | Less camera-aware; same-uniform occlusion still difficult | Required baseline |
| Segmentation/video foundation model + semantic identity | Masks can isolate uniforms and aid annotation | Additional compute, drift, and no automatic proof of persistent player identity | Optional experiment after baseline |

Roboflow's [American-football tutorial](https://blog.roboflow.com/american-football-player-tracker/) demonstrates RF-DETR Small with ByteTrack. It is a starting point, not evidence that IDs survive this clip's scrums. The newer [Roboflow trackers library](https://trackers.roboflow.com/) provides a cleaner integration path than assembling the original BoT-SORT research environment. Detailed capability and licensing evidence is in the [tool research](research/tracking-tools.md).

The verified release starting point is **RF-DETR 1.10.1, trackers 2.6.0, and supervision 0.30.2**. Lock these with a compatible Python/PyTorch environment after an actual installation and inference smoke test; the version snapshot alone does not prove compatibility. Prefer Small/Medium Apache-designated models for the baseline and check checkpoint terms separately.

For a modern second experiment, prioritize **McByte with mask assistance explicitly enabled** if the measured failures are crowded associations. Its default constructor leaves mask assistance off, and full mode needs SAM and Cutie. SAM 3.1 is another annotation/tracking experiment, but its published GPU throughput does not establish performance on this machine or accuracy for 22 football players. Neither alternative replaces the identity and replay layers. [Alternative evidence and setup requirements](research/tracking-tools.md#modern-alternatives)

## Processing architecture

```text
Local video
  -> decode with original frame indices and timestamps
  -> shot boundaries + field region + replay grouping
  -> RF-DETR detections, cached once
  -> BoT-SORT per shot -> immutable raw tracklets
  -> team evidence + crop gallery + optional jersey observations
  -> constrained identity matching <-> optional Astra evidence queue
  -> versioned player identities and unresolved cases

Field landmarks + camera motion -> time-varying calibration
Tracked ground-contact estimates + calibration -> field trajectories

Observations + identities + trajectories
  -> annotated video + field view + tables + review/evaluation report
```

Use two passes. Pass one produces local observations and short tracks. Pass two uses the full shot, including clearer post-play crops, to repair earlier identities. Keep both raw and resolved results so retrospective changes are visible. This offline mode deliberately permits future-frame evidence; do not report its scores as streaming performance.

### 1. Video and shot handling

Decode HEVC through FFmpeg/PyAV or a verified OpenCV backend. Keep integer presentation timestamps and time base; frame index is a separate key. Preserve output duration and audio alignment. For the first sample, process every source frame rather than discard half the available motion evidence before establishing accuracy.

Initialize two shot records using the verified cut, with a manual override supported. Later shot detection combines image change, feature-match collapse, and camera geometry discontinuity. Do not use the sample's scene threshold as a universal setting. Reset tracker motion state, active track IDs, camera compensation, image-space trails, and calibration on each cut.

Treat `shot_id`, `play_id`, and `player_id` as different concepts. Initially mark the second shot as a candidate replay of the first, then confirm through review. Use manually checked snap/action anchors to map media timestamps to play time. Allow piecewise alignment for repeats or changed playback speed. Until alignment is verified, export separate shot trajectories and leave play-time coordinates null.

### 2. Detection and region filtering

Start with RF-DETR Small as the cost baseline, then compare Medium at matched evaluation settings if distant-player recall is inadequate. Use a checkpoint trained for American football, or fine-tune one; generic COCO person weights are only a pipeline smoke test. The tutorial's `nfl-detection-1500-jdrgz/4` is a candidate to inspect, not a guaranteed downloadable or licensed local checkpoint. Validate export access, class mapping, and sample predictions before choosing it.

The detector taxonomy should distinguish `player` and `official`; collect `football` as a later separate task. Keep team as a track attribute so the detector can generalize beyond these two uniforms. If using an existing team-specific checkpoint, explicitly map its classes into this shared schema. Filter sideline personnel using a field polygon and role evidence; allow players to continue briefly into a boundary buffer after going out of bounds.

RF-DETR's NumPy input uses RGB. Decode/annotation may use BGR, so perform conversion only at the detector boundary and preserve original pixel coordinates. Keep low-confidence candidates for the tracker's second association stage instead of throwing them away using a high detector cutoff. Tune detection and association thresholds together on development footage.

For small players, first assess the model's input resizing. If needed, compare higher supported resolution and overlapping tiles, with duplicate suppression in original-image coordinates before tracking. Dense players must not be merged simply because their boxes overlap. Run inference as a bounded stream, not with every decoded frame resident in memory.

### 3. Per-shot tracking

Use `trackers.BoTSORTTracker` and compare `trackers.ByteTrackTracker` on exactly the same cached detections. Use `sv.Detections` as the adapter representation. The researched BoT-SORT update accepts detections, the frame for camera compensation, and timestamps. Confirm these signatures against the pinned release during implementation.

**Roboflow's current BoT-SORT variant does not include appearance ReID.** Its name must not be taken to imply that a player disappearing into a pile will automatically recover their identity. The original BoT-SORT paper/repository includes appearance-related machinery, but the recommended package requires our explicit identity layer. Also avoid starting new code around deprecated `supervision.ByteTrack`; use the separate `trackers` package. [Source details](research/tracking-tools.md)

Configure frame rate explicitly as 60000/1001 for this clip. Translate real-time gap durations into the library's buffer conventions: its lost-track buffer is expressed at a 30 fps reference rate and scaled internally. Do not scale the same buffer twice. Supply source timestamps consistently on every update; do not mix timestamped and untimestamped calls. Empty detections still advance tracking and expiry. A failed camera-motion estimate should lower matching confidence and use bounded motion-only association; it must not silently inject an arbitrary transform.

Keep a tracklet as one uninterrupted, locally plausible identity hypothesis. Add split points when team evidence conflicts, the motion jumps, or a crossing creates ambiguity; a long tracker ID is not proof that the same person occupied it throughout. Do not permanently delete raw tracker assignments when correcting a switch.

### 4. Team assignment and persistent identity

Collect several sharp, minimally occluded crops per tracklet. Estimate jersey color from torso regions, excluding turf, skin, and large background areas. Bootstrap two uniform groups and manually map them to DET/LAR once for this sample. Aggregate evidence through time; retain `unknown` for weak crops and separate officials from both teams. Color clustering alone distinguishes teams, not teammates.

Resolve identity using a graph of compatible tracklets. Candidate scores combine motion after camera compensation, valid field position, team evidence, and optionally jersey-number evidence and an evaluated sports appearance embedding. A general person-ReID checkpoint is an experiment because identical uniforms and helmets weaken its usefulness.

Within a shot, reject identity links between distinct simultaneously visible players, confidently different teams, or impossible travel. Score plausible time gaps with uncertainty that grows through occlusion. Solve competing matches jointly with one-to-one constraints; do not greedily let two tracks claim the same player. Establish match thresholds and a margin over the next candidate on development labels. Abstain if evidence is insufficient.

Across the two views, image-coordinate proximity has no meaning. Use verified replay alignment, field layout, formation relationships, and readable team/jersey evidence. Within synchronized play time, one identity may have an observation in each camera; the same-shot uniqueness rule still applies. Without sufficient cross-view evidence, preserve separate IDs and record a candidate relationship rather than asserting a merge.

Expose short-lived `tracklet_id` separately from stable `player_id`. Generate stable anonymous IDs deterministically from accepted identity groups. Save every split/merge with source observations and a decision version. Manual corrections are a separate overlay that can be reapplied after rerunning inference.

### 5. Optional GPT-6 Astra identity assistance

Use Astra for selected ambiguous tracklet matches and visible jersey/team evidence. Supply numbered crops from multiple timestamps, a wider context frame when useful, candidate IDs, and permitted decisions: `same`, `different`, or `insufficient_evidence`. Ask for supporting frame/crop IDs and readable number alternatives. Never ask it to invent missing player coordinates or resolve every unseen player.

The official [GPT-6 Astra model page](https://developers.openai.com/api/docs/models/gpt-6-astra) confirms image input, the Responses API, and Structured Outputs. Send sampled images and text rather than assume direct MP4 support. Use a strict response schema and validate the returned IDs against the supplied candidate set. Structured output constrains format, not factual correctness. [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

Use the local resolver to enforce physical and identity constraints after receiving a suggestion. Treat model confidence as an uncalibrated score, not a probability. Conflicting or unsupported answers go to review. Official vision documentation identifies small-text, spatial, and accuracy limitations that are directly relevant to distant jerseys. [Vision guide](https://developers.openai.com/api/docs/guides/images-vision)

The default first run is local-only. The optional API mode has explicit model selection, per-run budget, bounded request count, bounded retries, and request caching keyed by crop hashes, candidates, model, prompt, and schema versions. When disabled, unavailable, or over budget, finish local outputs and retain unresolved cases. Logs record requested/returned model, tokens, retries, wall time, and cost. No API call is required for the basic tracking deliverable.

### 6. Field trajectories

Produce pixel trajectories first and field trajectories only when calibration is valid. For this two-shot sample, manually label at least 6–8 well-distributed, non-collinear yard-line/hash/sideline intersections per calibration keyframe. Use additional withheld landmarks to test the fit. Fit a robust image-to-field homography and propagate between keyframes using static field features; refresh after pan/zoom drift. Camera compensation for tracking and metric field calibration solve different problems and have separate quality checks.

Define a fixed field coordinate frame in yards: x increases along the field from one chosen end line (0–120); y runs across its width (0–160/3). Record orientation explicitly and never redefine x by team possession or camera view. Goal lines are x=10 and x=110. Use NFL-specific hash geometry, not soccer or college templates. Field dimensions and markings are documented by [NFL Play Football](https://playfootball.nfl.com/tackle/youth-and-high-school-tackle-football-glossary/) and [NFL Football Operations](https://operations.nfl.com/the-rules/nfl-rulebook/).

Project a ground-contact estimate, initially bottom-center of the player box, not its center or helmet. Bottom-center is approximate when a player is crouched, airborne, tackled, or truncated; lower quality or leave field position null. Foot keypoints can be evaluated later if this error dominates. Never project an airborne ball through the same ground-plane mapping as though it were its true 3D position.

Store observed, predicted, and interpolated coordinates distinctly. Bound interpolation to short gaps with plausible endpoints; leave long occlusion/off-screen spans empty. Draw gaps honestly and do not sum them as measured distance. Render camera-independent paths on the field view. If rendering trails onto the moving video, reproject past field points through the current calibration; otherwise suppress trails during camera motion. Historical pixel coordinates drawn on a new camera view are misleading.

## Data and module contracts

Use a small Python package with a batch CLI first. The names below are proposed interfaces, not implemented files or executable commands.

| Module | Responsibility | Persisted contract |
| --- | --- | --- |
| `video` | Decode and identify shots/replays | frame index, PTS, time base, shot and play mapping |
| `detector` | Model loading, inference, coordinate conversion | boxes, scores, mapped classes, checkpoint hash |
| `tracking` | Per-shot tracker adapter | raw tracklets and observed/predicted state |
| `identity` | Team evidence and constrained matching | stable IDs, link evidence, uncertainty, corrections |
| `semantic` | Optional cropped-image requests | input hashes, structured suggestions, usage |
| `calibration` | Landmarks and transforms | per-time homography, field frame, validity/errors |
| `export` | Video, field view, tables | rendered outputs and common row schema |
| `evaluation` | Reference matching and metrics | automatic vs reviewed reports and failure clips |

Each observation includes `run_id`, `shot_id`, nullable `play_id`, `frame_index`, integer `pts`, `time_base`, nullable `play_time_s`, `tracklet_id`, nullable `player_id`, `bbox_xyxy_px`, `detection_score`, `team`, `team_score`, nullable `jersey_number`, nullable `field_xy_yards`, `position_source`, `calibration_id`, and `identity_version`. Coordinates remain in the original 1280 × 720 frame regardless of model resizing. Scores from different stages are stored separately.

Per-run outputs are `annotated.mp4`, `field-view.png`, `observations.csv`, `observations.parquet`, `trajectories.csv`, `identities.json`, `shots.json`, `calibration.json`, `review.json`, `metrics.json`, and `run-manifest.json`. `trajectories.csv` always has image-space contact points and leaves yard columns null until calibration is valid. Keep observations per shot. A separate play-aligned export may choose or fuse duplicate camera observations only after alignment is validated; preserve provenance and avoid double-counting distance.

The manifest records video hash, software versions, checkpoint hash/license, configuration hash, analysis mode, device, timing, and API usage. Cache detections separately from tracker output to make comparisons cheap. Distinguish `complete`, `complete_with_unresolved`, and `failed`; a missing detector checkpoint is a failure, not permission to render fabricated detections.

## Data strategy

Use Roboflow for dataset versioning, annotation, export, and RF-DETR training when appropriate. Before adopting a Universe dataset, inspect actual frames and classes: popular datasets called “football” are soccer. The candidate American-football tutorial and any public data are starting points whose export terms, licenses, and viewpoint match need checking.

For an initial sample adaptation, budget approximately 100–200 carefully chosen frames spanning both views, formation, crossings, contact, zoom, and post-play. This is a labeling estimate, not a sufficiency claim. Include officials and sideline negatives. Add persistent player-ID annotations separately; ordinary detection boxes are not tracking ground truth. Label visibility and ignore regions consistently.

Reserve the supplied clip primarily as a demonstration/development fixture. If it is used for training, both camera views of that same play belong in the same split. Test generalization on separately sourced games/plays, grouping all replay views together. Never random-split neighboring video frames. Do not mirror jersey-number training images unless number labels are transformed correctly; generic detector augmentations are not automatically suitable for OCR.

## Runtime and cost

The inspected development machine is Apple Silicon (M1 Max, 64 GB RAM). Plan an explicit MPS/CPU smoke test with the pinned RF-DETR/PyTorch build; select the supported device deliberately and record fallback. Do not copy CUDA/TensorRT latency claims onto this machine. Use an NVIDIA GPU later for training or throughput benchmarking if local measurements justify it. No hardware purchase or cloud job is needed to finalize this design.

At the user-quoted basketball rates, this clip would imply about **$4.75 and 380 seconds (6.34 minutes)** by simple linear extrapolation. That is a comparator only. Our actual cost depends on crop sizes, requests, token usage, caching, and retries; local inference also consumes compute. Measure decoding, detection, tracking, identity, API, and encoding separately. Report end-to-end wall seconds per source-video second and model-only throughput separately.

Astra's published image-sizing guide does not provide every Astra-specific pricing detail needed for a dependable per-crop quote. Use verified current rates and actual returned usage at execution time instead of applying another model family's token multiplier. An optional semantic pass should be evaluated against both the local baseline and a more frequent-call variant before claiming savings or quality improvements.

## Implementation sequence and exit criteria

1. **Media and baseline:** implement decode, shot overrides, detector adapter, cached detections, both tracker adapters, and basic video/table export. Process the entire supplied clip with no frame-order loss or cross-cut track carryover. A checkpoint must be available and class mapping verified.
2. **Stable IDs and teams:** add crop galleries, team aggregation, tracklet splits, constrained joining, and a correction file. Evaluate automatic output before applying corrections. Stable IDs must be supported by the [evaluation plan](evaluation-plan.md), not by visually persistent labels alone.
3. **Trajectories and replay handling:** calibrate both shots, export quality-aware field positions, verify replay alignment, and render field trajectories without duplicate play statistics. Until cross-view matches pass evaluation, ship them as unresolved while retaining valid shot-local output.
4. **Selective Astra experiment:** process a bounded queue of hard identity cases. Compare error reduction, coverage, latency, and spend against the same local run. Promote only if the benefit is measured.
5. **Later capabilities:** jersey-to-name lookup from a verified game-date roster, a small-object football detector, and a temporal possession model. Keep visible ball location, inferred carrier, ball-in-flight, contested, and unknown states separate. Nearest-player distance alone is insufficient for handoffs, passes, fakes, and piles.

The first three steps deliver the user's selected priorities; step four tests the closest analogue to the basketball system. Named recognition and possession do not block that deliverable. No claim of real-time speed, all-player identity recovery, or football-wide accuracy is made before the required measurements.
