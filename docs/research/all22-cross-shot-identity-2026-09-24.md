# All-22 cross-shot identity: automation research and execution options

Research date: **2026-09-24**. Repository inspected at `502115a`. This is a research
and design document, not an implementation or a new accuracy result. Read it with
the [implementation handoff](../goals/all22-identity-automation.md).

## Recommendation

**A largely automatic system is a reasonable engineering objective. The best next
investment is automatic field registration and replay synchronization feeding a
constrained identity resolver, with sports-specific visual cues and an optional
vision-language model (VLM) challenger.** Buying more inference alone will not
repair missing time maps, absent field coordinates, or contaminated local tracks.

Pursue three measured routes on the same footage:

1. **Local:** existing RF-DETR/BoT-SORT, NFL field geometry, automatic timing,
   tracklet repair, and uncertainty-aware assignment. This is the maintainable
   baseline and should remain useful with no external API.
2. **GPU server:** the same artifact contract, with learned field detection,
   stronger crop/jersey models, sports ReID, and optional synchronization models.
   A server supplies capacity; the algorithms and evidence still determine quality.
3. **Sparse Astra or another VLM:** automatically prepare short, source-addressed
   multi-view packets. Test both an independent direct-identity challenger and a
   geometry-assisted reranker. A model may request better crops or time windows;
   it must also be able to abstain.

For minimum personal workload, also evaluate a **commercial All-22 data service**
against the same acceptance set. This can outsource substantial engineering and
review, although it does not establish that the service itself is human-free.

The important change from the earlier research is to build toward **exception-only
review**, not perpetually require hand calibration and event marking for every
play. Keep a small, independently reviewed evaluation set, and progressively
replace operational manual steps with validated automatic predictors. Prediction
and evaluation truth need different provenance and readiness rules.

## Contents

- [What is actually blocked](#what-is-actually-blocked)
- [What identity means](#what-identity-means)
- [Options and priorities](#options-and-priorities)
- [Primary-source implementation shortlist](#primary-source-implementation-shortlist)
- [Automatic geometry and timing](#automatic-geometry-and-timing)
- [Player identity and local track quality](#player-identity-and-local-track-quality)
- [Astra and other VLM routes](#astra-and-other-vlm-routes)
- [Local and server execution](#local-and-server-execution)
- [Reducing manual intervention](#reducing-manual-intervention)
- [Experiments and acceptance](#experiments-and-acceptance)
- [Decisions and source limitations](#decisions-and-source-limitations)

## What is actually blocked

The [September 23 stage report](../evidence/all22-local-stage-2026-09-23.md)
records a 1,424-frame, 1280×720 sample at 60000/1001 FPS with the cut at frame 712.
These are historical measured results, read from the committed report; the
ignored source video, weights, and generated `artifacts/` are absent from this
checkout and were not rerun here.

| Recorded state | Meaning for this research |
| --- | --- |
| 18,023 observations; 57 shot-local tracklets | Not 57 different players; fragmentation and purity require measurement |
| Median 9 emitted boxes in sideline, 16 in end zone | Possible detection/filtering/association bottleneck; not measured recall because visibility is not labeled |
| Median 61 generic person proposals in sideline frames | Increasing detector sensitivity alone may include more sideline people; audit losses by pipeline stage |
| No yard coordinates; calibration `not_provided` | Geometric identity evidence does not exist yet |
| Identity `not_attempted`; evaluation `not_evaluated` | This is not a measured failure of an attempted cross-shot classifier |
| 255 unreviewed intersection hints; motion-burst proposals | Missing semantic yard labels and corresponding event identities |
| 296 Tesseract crops, no repeat-supported two-digit cross-shot match | This OCR configuration did not provide matching evidence; it does not rule out better crop selection or recognition |
| Two CVAT bundles; reviewed timing/calibration/reference missing | Source addressing is available; validation truth and operational automation remain open |

The stored [sideline frame](../evidence/sideline-1s.jpg) and
[end-zone frame](../evidence/endzone-14s.jpg) were visually inspected for this
research. They show complementary scale and visibility: more formation context
in the sideline frame, larger bodies/numbers but cropped edge players in the end
zone. This is qualitative inspection, not player counts, event alignment, or
identity annotation.

Current code already supplies useful infrastructure:

- [Replay alignment](../../src/football_tracking/replay.py) preserves PTS, fits
  bounded play-time maps, and checks withheld events. Its loader explicitly
  rejects alignment not marked `reviewed: true`.
- [Calibration timeline](../../src/football_tracking/calibration_timeline.py)
  tracks shot-specific fits, support intervals and polygons, and withheld quality.
- [Candidate scoring](../../src/football_tracking/replay.py) requires eligible
  compatible teams, at least five paired samples, at least 0.4 seconds overlap,
  and uses a 0.05-second pairing tolerance by default. Its current score combines
  field distance, trajectory shape, and team evidence. These defaults are
  implementation facts, **not validated operating thresholds for this clip**.
- [Identity cues](../../src/football_tracking/identity_cues.py) support jersey and
  embedding evidence, but automatic cues affect review ranking; reviewed reliable
  jersey conflicts constrain assignment. Adding a new embedding file alone does
  not make embeddings drive accepted automatic identities.
- [Resolution](../../src/football_tracking/identity_resolution.py) and
  [identity](../../src/football_tracking/identity.py) already support abstention,
  competing assignments and component consistency. Extend these seams before
  introducing a second independent identity system.

**Architectural implication:** create an explicitly versioned experimental
prediction path for machine-validated calibration/timing/identity. Do not pretend
automatic outputs are human-reviewed by setting `reviewed: true`. Preserve the
existing reviewed-reference loader and promotion gates. The current
[readiness script](../../scripts/check_identity_readiness.py) answers readiness
for **evaluation**, not whether every future production play needs human labels.

## What identity means

Separate four objectives; solve them in this order:

1. **Within-shot track purity:** an image track represents one physical player.
2. **Same-play cross-view identity:** a sideline track and end-zone replay track
   represent the same anonymous player during overlapping play time.
3. **Named identity:** team plus reliable jersey/roster evidence maps that player
   to a real roster entry.
4. **Across-play identity:** retain identity despite substitutions, similar
   formations, and changing visibility. This needs separate evaluation.

The immediate objective is (2). It can succeed without reading every jersey.
Synchronized ground position and motion are often stronger evidence than tiny
numbers. However, two visually indistinguishable teammates who enter a fully
occluded pile and emerge ambiguously may not be identifiable from the available
pixels. More model reasoning cannot guarantee recovery of missing information.
The product should return partial trajectories and unknown identity, with reasons.

Replays must share play time but retain distinct source observations. They must
not double-count distance or participation. All-22 describes the filming purpose;
it does not guarantee that all 22 players remain visible in both crops.

## Options and priorities

Ratings below are this research's engineering judgment. Published results on
soccer, basketball, pedestrians or synchronized feeds are not All-22 replay proof.

| Option | Role | Practical priority and limitation |
| --- | --- | --- |
| NFL line-grid registration plus bounded camera tracking | Automatic yards per view | **First:** use field topology and semantic anchors, not anonymous intersections |
| Motion/event-based time search | Automatic replay offset/rate | **First:** begin with simple bounded models and measured uncertainty |
| Joint field-space assignment with unmatched outcomes | Cross-view anonymous IDs | **First:** integrates existing repo primitives; needs usable local tracks |
| Football detector adaptation and track repair | Recover missing/polluted observations | **First if audit confirms loss:** more important than changing ReID blindly |
| Legibility-gated temporal jersey recognition | Strong intermittent identity cue | **Next:** quality selection and abstention matter more than OCRing every box |
| Sports ReID plus global ID memory | Rank lookalike candidates and reconnect fragments | **Next:** compare incremental benefit on same-team hard negatives |
| Sparse Astra direct comparison and assisted reranking | Semantic interpretation and ambiguity reduction | **Parallel bounded experiment:** useful even before geometric candidates exist, but remains unvalidated prediction |
| Open-weight VLM | Private/local or self-hosted crop reasoning | **Optional:** operational control, with workload-dependent memory and latency |
| VisualSync | Learned multi-view temporal-offset challenger | **Conditional server experiment:** expensive dependencies and replay-speed mismatch |
| DeepStream multi-view tracking | Production calibrated-camera framework | **Lower priority here:** camera-network machinery does not supply replay correspondence or NFL identity automatically |
| Commercial All-22 data service | Buy results and reduce personal review | **Viable alternative:** test custom-clip support, exports, errors, provenance and cost |
| RFID/Next Gen Stats identity tracks | External identity and position anchor | **Best extra signal if legitimately available:** a different input regime, not video-only inference |

## Primary-source implementation shortlist

Availability means public source/checkpoint links were observed on the research
date. Binaries were not downloaded, environments were not reproduced, and licenses
must be checked at the pinned revision, including dependencies and model weights.
The source-specific findings below are separated from the proposed architecture.

### Calibration and synchronization

| Method and source | What is available / demonstrated | Decision for All-22 |
| --- | --- | --- |
| **PnLCalib**, [paper](https://arxiv.org/abs/2404.08401), [authors' code](https://github.com/mguti97/PnLCalib) | Point/line detectors and nonlinear camera refinement; March 2026 update adds weighting and distortion optimization | Preferred learned-calibration architecture to adapt; soccer field semantics and weights need NFL replacement/validation |
| **BroadTrack**, WACV 2025 [paper](https://openaccess.thecvf.com/content/WACV2025/papers/Magera_BroadTrack_Broadcast_Camera_Tracking_for_Soccer_WACV_2025_paper.pdf), [code](https://github.com/evs-broadcast/BroadTrack) | Temporal camera/tripod modeling; documented CUDA/Docker route | Preferred temporal-camera reference; test physical assumptions separately for sideline and end-zone cameras |
| **TVCalib**, WACV 2023 [paper](https://arxiv.org/abs/2207.11709), [code](https://github.com/MM4SPA/tvcalib) | Segment-reprojection calibration, soccer segmentation and evaluation tooling | Useful line-based baseline; its self-verification does not replace independent absolute-yard checks |
| **American-football topology registration**, CVSports 2026 [abstract](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/html/Sawafuji_3D_Reconstruction_of_American_Football_Game_Situations_from_Handheld_Monocular_CVPRW_2026_paper.html) | Sawafuji et al. describe relative line-grid topology and physics-aware smoothing for handheld monocular football; explicitly qualitative validation | Strong domain-specific design lead, not ready weights or two-view identity proof; no official implementation was verified in this search |
| **SportsFields**, WACV 2021 [paper](https://openaccess.thecvf.com/content/WACV2021/html/Nie_A_Robust_and_Efficient_Framework_for_Sports-Field_Registration_WACV_2021_paper.html) | Grid keypoints and distance-map alignment across five sports including American football | Direct football precedent; usable author-released football weights were not verified |
| **KpSFR**, CVPRW 2022 [paper](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Chu_Sports_Field_Registration_via_Keypoints-Aware_Label_Condition_CVPRW_2022_paper.pdf), [code](https://github.com/ericsujw/KpSFR) | Soccer grid-keypoint models; MIT stated in README | Alternate detector experiment, lower priority than NFL topology plus point/line fitting |
| **VisualSync**, NeurIPS 2025 [paper](https://arxiv.org/html/2512.02017v1), [code](https://github.com/stevenlsw/visualsync) | Estimates cross-camera offsets through epipolar motion constraints; public stack uses VGGT, MASt3R, CoTracker3 and masks; CUDA 12.4 installation documented, code-cleanup caveat | Worth a bounded server experiment for constant-speed overlapping views; does not supply variable-speed replay maps or NFL validation |
| **SoccerNet replay grounding**, [official task](https://www.soccer-net.org/tasks/replay-grounding) | Coarse replay-to-main-event retrieval; tight tolerance evaluated over 1–5 seconds | Useful full-game pair proposal analogy, far too coarse as the final 50 ms player-alignment check |

VisualSync's paper reports dataset-dependent residuals; for example, EgoHumans
median 46.6 ms but mean 122.1 ms, and 3D-POP median 77.8 ms. Its published results
do not establish that it meets this repository's timing gate. Moving cameras,
repeated uniforms and narrow overlap must be tested rather than assumed handled.

The [universal sports-camera evaluation protocol](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Magera_A_Universal_Protocol_to_Benchmark_Camera_Calibration_for_Sports_CVPRW_2024_paper.pdf)
is useful for avoiding misleading field-overlap scores. Keep position errors and
valid coverage visible. Confirm NFL rather than college hash geometry using the
[official field rules](https://operations.nfl.com/the-rules/nfl-rulebook).

### ReID, jersey recognition and track repair

| Candidate and source | Availability / domain | Adoption decision |
| --- | --- | --- |
| **SportsReID OSNet**, [paper](https://arxiv.org/abs/2206.02373), [code/models](https://github.com/shallowlearn/sportsreid) | Sports centroid-loss training; SoccerNet checkpoint links including compact 2.2M-parameter OSNet; MIT code | First appearance baseline; isolate inference from old upstream environment and test same-team football pairs |
| **GTA-link**, [official code](https://github.com/sjc042/gta-link) | Offline single-camera splitting/linking and sports OSNet checkpoint path; MIT code | Track-purity/fragmentation comparator; do not apply its single-camera time assumptions directly across replay cuts |
| **PRTreID**, [paper](https://arxiv.org/abs/2401.09942), [code](https://github.com/VlSomers/prtreid) | Sports person/role/team modeling; SoccerNet integration/checkpoint reference; [license](https://github.com/VlSomers/prtreid/blob/main/LICENSE) is Hippocratic 3.0, not MIT | Stronger candidate after OSNet; review terms and actual part-visibility configuration before adoption |
| **CLIP-ReIdent**, [official code](https://github.com/KonradHabel/clip_reid) | Sports/basketball ReID implementation with checkpoint links; MIT code | Secondary appearance comparator; distinguish this specific trained method from generic CLIP cosine similarity |
| **Jersey-number pipeline**, [2024 paper](https://arxiv.org/abs/2405.13896), [code](https://github.com/mkoshkina/jersey-number-pipeline) | Legibility, crop selection, PARSeq and temporal aggregation; hockey/SoccerNet model links; CC BY-NC 3.0 pipeline | Useful full sports-OCR baseline; multiple dependencies and noncommercial terms must be addressed |
| **Uncertainty-aware JNR**, CVPRW 2025 [project](https://lukaszgrad.github.io/jnr/), [code/models](https://github.com/lukaszgrad/uncertainty-jnr) | Public SoccerNet ViT-S/B checkpoints and uncertainty-aware digit/tracklet predictions; CC BY-SA 4.0 code | Preferred focused jersey challenger; choose public weights and record the precise model rather than citing unavailable headline weights |
| **Global ID Fusion (GIF)**, WACV 2026 [paper](https://openaccess.thecvf.com/content/WACV2026/papers/Wojtulewicz_Advancing_Player_Identification_and_Tracking_with_Global_ID_Fusion_GIF_WACV_2026_paper.pdf), [code](https://github.com/Wojak27/GIF) | NBA multi-perspective linking and curated identity memory; September 2026 code/data update exists | Useful global-memory design; gallery dependence and NFL transfer prevent a zero-setup claim; identity-weight availability and license remain unresolved |
| **McByte++**, August 2026 [preprint](https://arxiv.org/abs/2608.15688), [code](https://github.com/tstanczyk95/McBytePlusPlus) | Mask/motion/appearance recovery; Apache-2.0 code; Linux CUDA installation | Optional within-shot challenger, not the replay solver; published H100 tracking throughput excludes detection, and frame-loading memory needs profiling |

Two reproducibility details matter. The current
[SoccerNet PRTreID configuration](https://github.com/SoccerNet/sn-gamestate/blob/main/sn_gamestate/configs/modules/reid/prtreid.yaml)
uses global embeddings and disables keypoint visibility weighting; choosing that
config does not implement a fully part-aware pipeline. The JNR project reports
83.52% for its public SoccerNet ViT-B, whereas its 85.62% headline uses a model
trained on a private 200M dataset with no checkpoint link in that table. Neither
score predicts NFL jersey accuracy.

GIF has actual implementation files despite stale “code soon” metadata, but the
examined README also contained merge markers. Its zero-shot approach uses a
curated identity gallery/attributes. Treat it as a reproduction project rather
than a turnkey package; no root license was observed and complete identity-weight
access was not established. Prefer sport-relevant team/number/equipment cues,
without importing unnecessary demographic classification features.

### Data and integrated-system evidence

**MOTAF / AFMOT (2025)** supplies a much closer research domain than pedestrian
ReID: university American football with synchronized handheld wide/tight views.
The [paper](https://arxiv.org/html/2511.09455v1) describes 10 plays, 20 videos and
11,411 annotated frames, and finds detector fine-tuning dominates its tracking
improvements. This is support for the detection audit, not an All-22 cross-shot
score. The [project](https://rinost081.github.io/MOTAF_page/) links a
[gated dataset](https://huggingface.co/datasets/rinost081/AFMOT) with noncommercial
academic conditions. Shared cross-view reference-ID semantics and football ReID
weights were not established here; verify before designing a training recipe.

**FieldMOT (2025)** registers observations before association, supporting a
field-space-tracking ablation. Its camera-switching sports evidence is synthetic
soccer, not NFL replay footage. See the
[primary paper](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Chen_FieldMOT_A_Field-Registered_Multi-Object_Tracking_for_Sports_Videos_CVPRW_2025_paper.pdf).

**SoccerNet game-state reconstruction** provides an integrated evaluation pattern
for position/team/role/jersey identity. The
[2025 challenge report](https://arxiv.org/html/2508.19182v1) includes systems
combining geometry, tracking, OSNet and VLM attribute prediction. That supports
testing hybrid cues, not importing soccer scores as NFL results. The
[2026 report](https://arxiv.org/html/2607.07320v1) includes different tasks such as
calibrated static-camera localization; task differences matter when comparing
newer “state of the art” claims.

**NFL helmet assignment** has excellent geometric lessons but a materially
different input. The [official competition data](https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/data)
includes 10 Hz tracking data, and the
[2023 homography-identification paper](https://openaccess.thecvf.com/content/CVPR2023W/CVSports/papers/Pandya_Homography_Based_Player_Identification_in_Live_Sports_CVPRW_2023_paper.pdf)
uses RFID-derived player positions. These solve matching against an external
identity anchor; they do not show that two anonymous videos alone provide it.

**General multi-camera frameworks and masks:**
[NVIDIA DeepStream MV3DT](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_MV3DT.html)
targets calibrated camera networks. It is relevant server infrastructure, but
adapting camera/time semantics for edited replays is additional work. An
[All-22 SAM3/SAM2.1 author's report](https://akhiltghosh.com/blog/all22-cv-pipeline)
demonstrates a manually selected player propagated within a shot, while explicitly
noting unresolved identification and occlusion issues. Masks can improve local
continuity; their persistence does not prove cross-shot identity.

## Automatic geometry and timing

### Field registration should exploit the NFL grid

**Proposed local baseline:** detect field lines and hash marks, classify line
families, enumerate plausible correspondences to the repository's canonical
[NFL field](../../src/football_tracking/field.py), and fit robust point/line
registration for each camera state. Preserve several candidates when repeated
five-yard lines or field reflection make the answer ambiguous.

Use visible numerals, midfield artwork, sidelines, end-zone boundaries, hash
spacing, and temporal continuity to disambiguate candidates. Yard-number OCR is
potentially easier than tiny jersey OCR, but a visible “40” does not identify
which half of the field it belongs to. Handle rotation and perspective. Neither
player formation nor a low line-fit residual alone establishes the absolute
field origin. If only relative geometry is identifiable, report it as such;
absolute yard calibration remains unresolved.

Choose keyframes automatically when camera motion, residuals or support change.
Mask players, officials, score graphics and moving camera rigs during field-motion
estimation. Track stationary field evidence between absolute fits, periodically
relocalize, and stop propagation at cuts or unsupported intervals. Never reuse one
view's homography in the other, or linearly interpolate arbitrary homography
matrix entries. A pan/zoom model may reduce degrees of freedom when its physical
camera assumptions actually fit.

### Withheld checks have two distinct meanings

At development/evaluation time, use independently labeled semantic landmarks
withheld from fitting. A model cannot certify absolute accuracy against landmarks
whose coordinates it invented itself. Split checks by spatial region and camera
state; reserve more than one when available. The repo's one-check minimum can
exercise a gate, but cannot characterize a p95 distribution or distinguish many
systematic errors.

At inference time, use automatic consistency checks: held-out detected markings,
template topology, fit conditioning, support coverage, cycle consistency,
alternative-hypothesis margin, and drift. Call these **internal checks**, not
independent ground truth. Their ability to predict actual error must be calibrated
on the independent development set. A five-yard-shifted template may pass many
internal residual tests; the uncertainty must include discrete semantic ambiguity.

### Ground contact is separate from field calibration

Project the player's ground contact, not helmet or box center. Begin with the
existing box-bottom proxy and quality flags; test feet/pose/mask support only if
contact error dominates. Occluded feet, airborne players, leaning bodies and
cropped boxes can make ground position unknown despite an excellent field fit.
Do not use a planar homography on helmet coordinates without a height model.

### Align the underlying action, not the file timestamp

**Proposed automatic timing baseline:**

1. Propose same-play pairs from shot metadata, formation context and event order.
   Include wrong-play negatives; adjacent shots are only candidates.
2. Estimate camera-compensated player motion and coarse event windows. Existing
   motion bursts are inputs; they are not labeled snaps.
3. Search a bounded offset and speed ratio using view-invariant signals: team
   occupancy in field space, velocity magnitudes, motion onset and interactions.
   Compare unordered player sets before identities are known.
4. Refine on multiple events and aligned trajectories; include unmatched players
   and robust trimming so partial fields of view do not dominate the fit.
5. Use an affine map for constant-speed playback; only add bounded piecewise
   mappings or constrained dynamic time warping when evidence requires them.
   Detect freezes, duplicate frames and cuts. Unsupported spans remain invalid.
6. Validate on events excluded from fitting and on a locked set of plays. Preserve
   both original integer PTS and mapped play time, including residual uncertainty.

Avoid circular success: choosing the time map solely because a proposed identity
assignment matches, then treating that match as independent identity evidence,
can produce a confident wrong solution. Keep field/event evidence independent
where possible, retain alternate timing hypotheses, and evaluate joint fitting
on unused events. A VLM's coarse event timestamp can seed a local native-frame
search; it is not a 50 ms timing measurement.

### Joint inference after the baseline

The difficult variables interact: camera mapping, time mapping, track fragments
and identity. Start with modular fits for diagnosis, then test a bounded
alternating optimizer: field/time hypotheses → identity assignment → restricted
refinement → independent checks. Use beam search over a small number of plausible
orientation/yard-offset/time hypotheses instead of committing too early.

For candidate tracks A and B, a useful **proposed** geometric term is a robust
aggregate of `r(t)^T S(t)^-1 r(t)`, where `r` is their aligned field-position
difference and `S` includes calibration, contact and timing-induced uncertainty.
Timing contributes approximately `v v^T sigma_t^2` for a local velocity `v`;
correlated camera errors need separate treatment rather than assuming every
frame is independent. Penalize missing evidence and insufficient support. A
large uncertainty must not make every pairing appear acceptable.

Use team, jersey distributions, appearance, relative neighbors and trajectory
shape as separate features. Formation roles and routes are soft priors: two
linemen or crossing receivers must not be assigned by expected role alone.
Retain one-to-one assignment per simultaneous view interval and explicit
unmatched choices. Support multiple nonoverlapping fragments of one identity,
while rejecting components containing conflicting simultaneous players.

## Player identity and local track quality

### Audit detections before training a larger identity model

Instrument the journey from raw person detection → filtering → association →
export for each view. On a small labeled set, distinguish missed on-field players,
sideline people, officials, low-confidence tracks, lost associations and mixed
tracklets. Compare detector size/resolution/tiling only in a separate detection
experiment; compare trackers on identical detection caches.

Split a tracklet at credible appearance/team/jersey/trajectory discontinuities,
retaining raw rows and source lineage. Cross-view matching of a mixed tracklet
can contaminate every global component it touches. Tracklet purity, fragmentation
and recovery through blocks/piles matter more than simply reducing ID count.

### Replace indiscriminate OCR with a temporal recognition pipeline

Select nonredundant crops across each tracklet: size, sharpness, torso visibility,
view angle and occlusion. Detect/localize the jersey region, predict legibility,
then recognize a distribution over complete numbers and an explicit unreadable
state. Aggregate a few distinct views, rather than treating 100 neighboring
near-identical frames as 100 independent votes. Preserve disagreement as evidence
of bad reads or a contaminated tracklet.

Compare a sports-trained recognizer against the existing Tesseract baseline and
VLM crop readings. Do not turn a single digit into a full jersey number or force
roster-consistent guesses. Team + number can support a named-player lookup only
with game-specific roster provenance and readable evidence. Generative
super-resolution may fabricate a digit: enhancement can propose a crop for
inspection, but cannot create independent identity evidence.

### Appearance works best as an additional cue

Uniform color helps distinguish teams; it often does little for teammates.
Part-aware embeddings, body proportions, equipment details, jersey regions and
temporal appearance galleries may help, but camera angle, pads and occlusion
change them substantially. Train/evaluate with same-team, adjacent-player hard
negatives and cross-view positives, not only easy opposite-team pairs.

Keep embedding model/version and crop transforms in the cue artifact. Do not
compare vectors from unrelated models as if they shared a metric space. Evaluate
tracklet-level retrieval and downstream false merges separately. A high retrieval
rank is not a probability, and a pairwise improvement can still create a bad
transitive identity component.

## Astra and other VLM routes

### What submitting work to Astra actually means

Official [GPT-6 Astra documentation](https://developers.openai.com/api/docs/models/gpt-6-astra)
lists text and image input, text output, structured outputs, function calling and
MCP support. It does not list native video input. There are two useful submission
patterns:

- **Vision request:** send selected images/crops and source metadata to the model;
  receive structured observations or identity proposals.
- **Execution request:** give the model access to a controlled tool that submits
  the actual computer-vision job to your CPU/GPU worker, polls its status and
  reads artifacts. The external worker runs the pipeline. An Astra subscription
  or request by itself does not establish access to your chosen CUDA hardware,
  footage, checkpoints or job environment.

The [OpenAI MCP guide](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)
documents external service tools and private-server connectivity. A normal CLI or
HTTP client should also be able to submit the exact same job; an LLM is optional
orchestration, not a required component of every batch.

This research is the Astra/research role requested by the user. The accompanying
handoff delegates implementation and rollout to another coding model; no remote
execution, uploads or API inference were performed for this document.

### Test Astra in two roles, not only after the current empty queue

**Independent challenger:** construct packets from shot-local tracks even when
geometry is missing. Supply both view overviews, multi-frame crops of the target,
several plausible candidates, and a none-of-these option. Ask for same/different/
insufficient-evidence, jersey observations, coarse event correspondences and
specific evidence IDs. Keep this output in a proposal artifact independent of
the production resolver. This gives a direct test of whether the model can reduce
the initial manual bottleneck.

**Assisted resolver:** use validated geometry/timing to shortlist candidates and
provide aligned context. Test whether the model fixes residual ambiguities without
increasing false merges. Compare model decisions blinded to the local score
against decisions shown that score, to expose anchoring. Repeated calls or two
models agreeing are not independent human reference labels.

Roboflow's [September 18 Astra experiments](https://blog.roboflow.com/gpt-6-astra-vision/)
demonstrate sampled basketball re-identification and propose combining sparse
model interpretation with local detection/tracking. They do not report an NFL
All-22 cross-shot benchmark. Treat this as a concrete experimental precedent,
not a claim that this clip should already be solvable by one prompt.

### Proposed packet and response contract

For every packet, preserve source SHA-256, play/shot/tracklet IDs, original frame,
integer PTS, time base, crop box, crop-to-source transform, image dimensions and
actual sampling policy. Include full-view context plus readable separate crops;
avoid compressing 22 players into one tiny contact sheet. Add labels outside
jersey regions. Supply timestamps in text: the
[vision guide](https://developers.openai.com/api/docs/guides/images-vision) warns
about small text, spatial precision, metadata and image resizing.

Start with two to four temporally distinct crops per candidate and short event
windows; these are experimental packet settings, not proven optima. Let the
model request a specific original-frame crop if a number or interaction is
unclear. Bound requests, image count, output tokens, retries and total spend.

Required response fields, to be implemented as a strict schema:

| Field | Purpose |
| --- | --- |
| Packet/source/analysis IDs | Reject stale or mismatched responses |
| Target tracklet; candidate tracklet or null | Refer only to supplied IDs |
| `same`, `different`, `insufficient_evidence` | No forced complete assignment |
| Evidence image/frame IDs | Every factual observation traceable to pixels |
| Observed number distribution; legibility | Unknown is allowed; no inferred digits |
| Contradictions and missing evidence | Expose why a link may be wrong |
| Optional requested next crops/windows | Bounded active inspection |
| Provider/model/request/usage metadata | Reproducibility and cost accounting |

Any model confidence is an uncalibrated score until measured. Validate schema,
IDs, referenced frames, coordinates and constraints locally; never execute model
text as code. Refusal, timeout, invalid JSON, budget exhaustion and unavailable
provider must produce abstention and leave valid local results intact.

### Other hosted and open-weight choices

| Candidate | Why test it | Boundary |
| --- | --- | --- |
| [Gemini video API](https://ai.google.dev/gemini-api/docs/video-understanding) | Accepts video; useful for play grouping and event-window proposals; documented static and adaptive processing options | Default static sampling is 1 FPS, which can miss short actions; video support is not native-frame identity accuracy |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Downloadable Apache-2.0 model card; practical open-weight image/video reasoning comparator on suitable local/server hardware | No checked All-22 identity result; quantization and context affect memory and quality; pin dependencies rather than copying stale installation snippets |
| Specialized sports jersey/ReID models | Narrow outputs, easier batching and potentially lower cost | Need football transfer tests and checkpoint/license verification |

Use one alternative VLM initially, not a broad paid tournament. Compare the same
frozen packets, source resolution and abstention policy. A larger context window
does not restore detail lost during sampling or resizing.

## Local and server execution

### Deployment choices

| Route | Suitable work | Recommendation |
| --- | --- | --- |
| Existing local CPU / available accelerator | Decode, caches, line fitting, assignment, reports and bounded existing detector runs | Default baseline; measure effective device and peak memory |
| Local NVIDIA GPU or owned Linux server | Learned field detector, crop embeddings, jersey batches, masks | Prefer if already available; run identical versioned job payloads |
| [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs) | One-off Python/container CPU/GPU experiments and fine-tuning | Good first batch option when using Hub artifacts; it runs code, not a prebuilt All-22 identity model |
| [Modal GPU functions](https://modal.com/docs/guide/gpu) and [job queues](https://modal.com/docs/guide/job-queue) | Python pipeline with asynchronous submit/poll and reusable model loading | Good alternative when building a service; profile cold starts and persistence |
| [Runpod queue endpoints](https://docs.runpod.io/serverless/endpoints/send-requests) or Pods | Containerized GPU worker or interactive development machine | Useful for CUDA stacks; explicitly persist outputs and constrain concurrency/retries |
| Astra → tool → one of these workers | Natural-language job control and artifact interpretation | Add after a plain CLI job works; keep scheduling deterministic |

These are alternative hosts, not three integrations to build at once. Start with
local execution, then one server adapter selected for existing account/access.
No account availability or server runtime was verified during this research.

For planning only, a 24–48 GB NVIDIA worker is a reasonable first profiling target
for sequential detector/jersey/ReID experiments; it is **not a verified minimum**.
Large multi-view feature stacks or VLM contexts may need more. Load models
sequentially, limit decoded-frame buffers, and measure VRAM before scaling. Do not
assume CUDA research code works on Apple MPS or within the repo's 2 GiB default
RSS budget. Preserve explicit opt-in for heavy backends.

### Proposed portable job contract

An immutable job contains a source URI and hash, explicit source-frame/PTS ranges,
play/view metadata, cache hashes, pipeline revision, model/checkpoint hashes,
configuration, permitted execution backend, resource/time/spend caps and output
destination. Source bytes or lossless source-mapped extracts must be available to
the worker; a local pathname in a prompt is not a server upload.

Worker stages: validate input → decode/cache → per-shot observations → geometry
and timing proposals → local/cross-view inference → validation → artifact export.
Resume by content hash, write partial stage results atomically, and never combine
outputs from incompatible checkpoints or source scopes.

Proposed control interface: `submit_identity_job`, `get_identity_job_status`,
`cancel_identity_job`, `get_identity_job_artifacts`. These names are **new API
designs**, not callable tools already present in the repository. Submissions use
an idempotency key; retries cannot create duplicate billable jobs. Artifacts must
survive worker shutdown. A successful job execution can still report unresolved
identity or unevaluated accuracy.

### Cost model

Verified [OpenAI pricing](https://developers.openai.com/api/docs/pricing) on the
research date lists Astra Standard short-context input at $10 per million tokens
and output at $50 per million. Batch rates are $5/$25. Cache and long-context
rates differ; the [model page](https://developers.openai.com/api/docs/models/gpt-6-astra)
documents the greater-than-272K input boundary. Recheck rates before execution.

For an uncached short-context request, illustrative cost is
`10 * input_tokens / 1e6 + 50 * output_tokens / 1e6`.
Thus 20,000 total input tokens and 2,000 billable output tokens would cost $0.30
at those Standard rates, excluding tools and other charges. This is arithmetic,
not a measured packet size, quality result or quote for the clip. Include images,
reasoning/output usage, retries and cache operations in actual accounting.

For a GPU experiment, record billed compute seconds, rate, startup/model loading,
CPU/RAM/storage/egress and retries. Compare **cost per correctly resolved shared
player** and **human minutes saved**, not just price per frame. Do not estimate a
whole season from an unrelated basketball demo or extrapolate throughput before
a bounded same-pipeline trial.

### Commercial alternatives

[SkillCorner](https://www.skillcorner.com/sports/american-football) advertises
All-22-derived XY tracking and American-football data delivery through an API,
including NFL and college coverage. This is a serious buy-versus-build candidate.
Public material does not establish self-service upload of this particular clip,
per-link confidence, this task's accuracy or a public price. Request a sample
delivery and separate observed positions from predicted/off-camera positions.

[Hudl IQ describes](https://www.hudl.com/blog/how-hudl-iq-creates-football-data)
video-based field registration and player tracking with a hybrid computer/human
workflow, including identity and crowded-track correction. That is direct
evidence that substantial automation and residual manual work can coexist in
football production systems. Buying such a service may minimize the user's own
intervention even when the vendor performs review.

For either vendor, evaluate a fixed sample with sideline/end-zone replay pairs:
anonymous and named IDs, source-frame traceability, calibration/timing quality,
visible versus inferred positions, error corrections, custom-source support,
turnaround, exports and full cost. No vendor was contacted or footage submitted.

## Reducing manual intervention

### Separate development investment from recurring operation

The intended end state is: provide footage → receive trajectories and IDs →
inspect only a small exception queue. A reference set is still needed to know
whether automatic acceptance is reliable. Do not confuse that one-time
measurement cost with a requirement to label every future play.

| Phase | Human work | Automation to build |
| --- | --- | --- |
| Bootstrap | Small independent geometry/timing/identity sample; second review for cross-shot truth | Proposals, synchronized review, active crop selection and correction capture |
| Shadow operation | Review measured samples and exceptions; automatic outputs not yet promoted | Full unattended pipeline, failure categorization and calibrated acceptance policy |
| Selective production | Exception queue plus random accepted-case audits | Automatically accept supported cases and abstain on the rest |
| Expansion | Audit new camera/uniform/stadium conditions | Drift checks; active selection for targeted retraining |

The existing [review contract](../human-review-all22.md) requires independent
second review of cross-shot reference labels. Neither Astra nor another model
should invent that sign-off. Automatic predictions can be generated while the
reference is being prepared; evaluation and promotion wait for the actual labels.

### Review the highest-value uncertainty

A single correction to a wrong yard-line offset or replay speed can improve many
player matches. Prioritize play grouping, orientation/yard semantics, camera
drift and time maps before asking for dozens of pairwise IDs. Then review mixed
tracklets, same-team near ties, occlusions and unreadable crops. Show synchronized
views with source-frame navigation, alternatives and an unknown button.

Use active learning to select informative training examples, but preserve an
untouched evaluation set and randomly audit some confident predictions. Reviewing
only model-selected difficult cases cannot estimate the error rate of accepted
cases. Add corrected training labels with provenance; prevent pseudo-labels from
silently becoming the sealed test truth.

### Suggested labor goals

These are proposed development goals, not capabilities already achieved:

- Reduce median correction/review minutes per play by at least 50% against the
  same reviewers' measured manual baseline, without relaxing false-merge gates.
- After bootstrap, seek at least 80% of supported-domain plays processed without
  user intervention; report unresolved IDs on those plays separately.
- Report p50/p95 intervention time, number of clicks/corrections, fraction of
  plays needing review, and identity coverage at each automatic-acceptance setting.

“No review” is not success if the system abstains on every player. Report labor,
coverage and accuracy together. Maintain separate outputs for fully automatic,
automatic plus VLM, and human-corrected modes.

## Experiments and acceptance

### Start with a bottleneck ladder

Build one independent reference slice around a common visible action interval,
then expand to complete paired shots. Run:

| Experiment | Fixed inputs | Question |
| --- | --- | --- |
| Oracle geometry/timing + reviewed local tracks | Gold inputs; no learned ReID | Is shared-player identity identifiable by geometry here? |
| Oracle geometry/timing + existing tracks | Same identities and evaluation span | How much do detection/track purity errors cost? |
| Automatic geometry + oracle timing/tracks | Same reference | Does registration fail on yard semantics, drift or contact? |
| Automatic timing + oracle geometry/tracks | Same reference | Is temporal uncertainty the bottleneck? |
| Fully automatic geometry/team resolver | Same frozen detector cache | What is the deployable local baseline? |
| Add sports jersey; add ReID; add both | Same candidates and split | Does each cue improve coverage at the same false-merge constraint? |
| Astra direct and geometry-assisted variants | Frozen packets; source-matched truth | Does VLM interpretation save review or recover identities? |
| Local versus selected GPU worker | Same config, models and source hash | Does hosting improve practical throughput without changing semantics? |

Oracle results diagnose missing capability; they are never advertised as automatic
performance. Include partial/no-overlap, wrong-play pairs, wrong five-yard offsets,
mirrored fields, slow motion, freeze frames, camera pans, lineman collisions,
readable/unreadable numbers and absent players. Add known timestamp/geometry
perturbations to verify that the pipeline rejects attractive but wrong matches.

### Preserve existing accuracy gates

The [evaluation plan](../evaluation-plan.md) remains authoritative:

| Measure | Existing target / required reporting |
| --- | --- |
| Cross-view identities | Zero observed false merges on the sample and at least 0.80 coverage of independently human-resolvable shared players |
| Field calibration | Withheld median at most 1 yard; p95 at most 2 yards, stratified by view and camera state |
| Ground contact | Median at most 1.5 yards; valid output on at least 0.90 of evaluable observations |
| Within-shot tracking | IDF1 at least 0.90 and at most one ID switch per shot on sample; report HOTA/DetA/AssA and fragmentation |
| Detection | Visible-player precision and recall at least 0.95 at IoU 0.5, by view |
| Timing | Preserve the loader's default 50 ms maximum withheld residual policy unless changed prospectively with evidence; report support and uncertainty |
| Reliability | Useful local output with API disabled, failure, timeout or exhausted budget |

Some targets may prove unsuitable for the source quality. Report the failure and
propose a prospective policy revision; do not retroactively loosen the gate to
declare the first experiment successful. Geometry thresholds can be acceptable
for trajectory display yet too loose to separate packed linemen: candidate margin
and local uncertainty still matter.

Report exact correct/incorrect/missing/ambiguous counts, coverage denominator,
component contamination, and purity per physical player. Five repeated crops
are not five independent identity trials. Zero errors on one play says little
about rare failures. For intuition only, with independent Bernoulli trials and
zero errors, a one-sided 95% upper bound is `1 - 0.05^(1/n)`; at `n=22` this is
about 12.7%, and around 300 independent trials are needed to get below 1%.
Football links are correlated within plays, so report play/game clusters and
broader test coverage instead of treating this calculation as a guarantee.

Use the whole short clip for development. Keep all views of a play in one split;
do not call a random-frame split held-out generalization. The repo requires at
least ten additional plays from at least three games as an initial breadth gate.
A single full game allows useful within-game testing, but cannot satisfy that
cross-game gate. Freeze thresholds before held-out evaluation.

## Decisions and source limitations

**Build first:** diagnostic coverage/purity report, automatic NFL geometry and
time proposals, bounded machine-inference lane, existing constrained resolver,
and a compact exception review flow. In parallel, prepare the sparse VLM
challenger so the Astra hypothesis receives a fair measured test.

**Adopt only after comparison:** specialized jersey recognition, sports ReID,
mask-assisted tracking, learned temporal synchronization, or a more complex joint
optimizer. Train the component implicated by the bottleneck ladder, rather than
fine-tune every model at once.

**Defer:** whole-game named identities, unconditional complete assignment,
3D reconstruction as a prerequisite, new tracker defaults, and large-scale
server spend before a bounded experiment establishes value.

This pass uses current official model/provider documentation, author code/model
pages and primary papers, plus explicitly labeled first-party demonstrations.
It does not establish a universal state of the art or prove absence of other
solutions. Public code availability is not dependency reproduction, license
clearance, checkpoint compatibility or success on this source.

GitHub CLI authentication failed in the delegated source checks; public official
repository pages were used as the read-only fallback. Some publisher PDF access
failed: the 2026 topology-registration finding is limited to its indexed primary
abstract, and no implementation was verified. This does not prove none exists.
No new checkpoint, dataset, training job, CVAT server, identity API call or remote
deployment was executed. The next model should pin and reproduce only the selected
shortlist, then preserve these limitations in its result report.
