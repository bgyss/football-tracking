# Football Tracking: Social Posts

The baseline campaign was drafted September 10, 2026. The McByte follow-up campaign was drafted September 14, 2026. Each platform section contains ready-to-copy post text. The long-term goal is full reconstruction of football plays from All-22 footage.

## CVAT and Identity: Automating the Learning Loop

Drafted September 23, 2026 against local commit `a0d7333`. Copy below is ready to
paste; attachment instructions and claim notes are separate from the posts.

### LinkedIn: The Learning Loop

I'm working toward completely automating the fine-tuning loop for my All-22 football tracking project, with a bigger goal: processing game film faster and reconstructing plays more accurately.

The current focus is CVAT and player identity. Football makes both difficult: distant players, nearly identical uniforms, crowded contact, and a sideline view followed by an end-zone replay of the same action.

I've added a local CVAT import/export bridge so model-generated tracks can be corrected and brought back into the pipeline. Those corrections keep their original source frames, timestamps, and model-output provenance. The identity tooling also builds review queues for candidate matches across views, with uncertain cases allowed to remain unresolved.

The loop I'm building toward is:

Process footage → surface difficult cases → review and preserve corrections → fine-tune → evaluate on held-out games → repeat.

The aim is to make each round of review produce reusable training data and reduce the manual effort required for the next game. Better detection is one part of that; keeping the same player identity through contact and across replay angles needs its own evaluation.

Human review is still part of the workflow today. End-to-end retraining automation and measured speed and accuracy gains are the next steps, rather than results I'm claiming already.

Ultimately, I want to turn All-22 into a reliable account of who moved where, when, and how the views fit together.

Project: https://github.com/bgyss/football-tracking

#ComputerVision #FootballAnalytics #MachineLearning

### Reddit: Technical Discussion

#### Title

Building toward an automated All-22 fine-tuning loop: CVAT corrections and cross-view player identity

#### Body

I'm building an offline All-22 football tracking project. The long-term goal is to completely automate the fine-tuning process so I can process footage faster and reconstruct plays more accurately. Right now, I'm working on the annotation and identity pieces that make that loop possible.

The baseline uses Roboflow's RF-DETR for detection and BoT-SORT for tracking. The difficult part is turning those outputs into useful supervision without carrying the model's mistakes straight into the next training set.

The latest work includes:

- A local CVAT video XML bridge for exporting proposal tracks and importing corrections. A source-hashed frame map preserves original frame numbers, timestamps, and crop coordinates; raw inference provenance stays attached.
- Reviewed annotation manifests and a player-only MOT reference export. Imports stay unreviewed by default, and the reference export requires explicit review.
- Identity review queues containing candidate links, competing matches, and unmatched tracks, addressed back to the original footage.
- Cross-view identity logic using validated field calibration, aligned play time, team evidence, and trajectory information. It can abstain when the evidence is insufficient.
- Optional jersey-number proposals to help rank review cases. Automatic jersey reads do not establish identity by themselves.

A sideline shot and an end-zone replay can show the same play at different video times. Treating them as one continuous track would create a misleading reconstruction. The aim is to align them in play time and field coordinates before deciding which tracks belong to the same player.

The intended learning loop is: run inference, prioritize difficult cases, review corrections in CVAT, curate detector-training labels, fine-tune, and evaluate on held-out games before adopting a new checkpoint. Both views of a play need to stay in the same data split.

I'm keeping detector labels and identity references separate. Better boxes do not automatically prove fewer identity switches. Detection quality, identity continuity, cross-view matching, and processing cost each need their own measurements.

Current status: the local interchange and identity-review tooling is implemented. Fully automated retraining is still the goal; human review remains necessary, and I don't yet have a held-out result showing improved reconstruction accuracy or throughput. The attached diagrams illustrate the workflow, not benchmark results or a CVAT UI session.

Repo: https://github.com/bgyss/football-tracking

For people working on sports video or active learning: how do you choose which crowded-contact or cross-view identity cases are worth annotating next?

### X: Standalone Post

I'm working toward fully automating fine-tuning for All-22: CVAT corrections, player identity, retraining, evaluation. The goal: faster film processing and more accurate play reconstruction. Human review is still part of the loop. https://github.com/bgyss/football-tracking

### X: Optional Four-Post Thread

#### 1

I'm building toward an automated fine-tuning loop for All-22 football footage. The goal: process film faster and reconstruct plays more accurately. The current work is CVAT + player identity. https://github.com/bgyss/football-tracking

#### 2

The CVAT bridge exports proposal tracks and imports corrections while preserving original frames, timestamps, and model provenance. Imports stay unreviewed by default. Identity review queues help surface candidate matches and competing explanations.

#### 3

Same player, different camera, different video time. Cross-view identity needs field calibration and aligned play time. Jersey reads can help rank review cases, but uncertain matches stay unresolved. Better boxes alone don't prove better identity.

#### 4

The loop I'm aiming for: inference → hard cases → reviewed labels → fine-tuning → held-out evaluation → repeat. Human review remains today. Automated retraining and measured speed/accuracy gains are still ahead. https://github.com/bgyss/football-tracking

### Attachments and Alt Text

Use the learning-loop graphic first on LinkedIn and with the standalone X post.
Use both graphics for Reddit, or attach the identity graphic to post 3 of the X thread.
The PNGs are upload-ready 1600 × 1000 images; the SVGs are editable originals.

- **Learning loop:** [PNG](social/2026-09-23-cvat-identity/learning-loop.png) · [SVG](social/2026-09-23-cvat-identity/learning-loop.svg).
  Alt text: Diagram of an All-22 learning loop: footage, detection and tracking, human CVAT review, preserved reviewed labels, planned detector fine-tuning, and planned held-out evaluation. Implemented tooling, human review, and future automation are labeled separately. Detection boxes train the detector; persistent tracks test identity.
- **Cross-view identity:** [PNG](social/2026-09-23-cvat-identity/cross-view-identity.png) · [SVG](social/2026-09-23-cvat-identity/cross-view-identity.svg).
  Alt text: Schematic of sideline and end-zone views mapping into shared field coordinates at aligned play time. Candidate player tracks are compared using geometry, timing, team, and motion. Automatic jersey cues help review; weak evidence leaves identity unresolved. The positions are illustrative, not measured tracking results.

### Editorial Claim Notes — Do Not Paste

- Verified against the current checkout: [CVAT workflow](annotation-and-data-generation.md), [import script](../scripts/import_cvat.py), [CVAT source](../src/football_tracking/cvat.py), [identity resolver](../src/football_tracking/identity_resolution.py), and [review queue](../src/football_tracking/identity_review.py).
- The [September 22 implementation note](evidence/cross-shot-identity-automation-research-2026-09-22.md#implementation-status) records generated-video interface checks, not full-game labeling or a live CVAT UI round-trip. No new model run, human annotation, or accuracy benchmark was performed for this campaign.
- Complete retraining orchestration, reduced annotation effort, faster processing, and better reconstruction accuracy are goals. These drafts do not claim a measured improvement or a completed autonomous training system.
- Reconstruction here means player identities and field trajectories across views; full play reconstruction remains the broader goal. No completed 3D reconstruction, named-player recognition, or ball-possession result is implied.
- CVAT is the annotation integration and Roboflow RF-DETR is the detector; these posts describe this project's integration and workflow, not authorship of those upstream tools.
- Both visual aids are original vector schematics. They contain no game footage, measured trajectories, benchmark values, or simulated CVAT screenshots.
- X drafts are checked against the [standard 280-character post limit](https://help.x.com/en/using-x/how-to-post), including the full written repository URL as a conservative count.

## McByte Follow-Up Campaign

This campaign is a three-part sequence: the integration milestone, the identity-verification gate, and the football-specific fine-tuning loop. The posts deliberately distinguish a working mask-assisted run from measured identity accuracy.

### LinkedIn Post 1: McByte Is Running

I just added an opt-in McByte tracking path to my All-22 football tracking project.

McByte adds propagated player masks to box-based association. The idea is especially interesting for football, where players overlap, cross, and disappear into contact while wearing nearly identical uniforms.

The integration now runs locally on the full 1,424-frame sample using RF-DETR detections, SAM, and Cutie. The pipeline preserves every source frame, resets tracking state at the camera cut, records whether masks remained active, and can replay the same frozen detections across trackers for a fair comparison.

The first full mask-active run completed without mask fallback. It also peaked at roughly 20 GB of host memory and took about 63 minutes for 23.76 seconds of video on the current Mac. That makes it a verified integration, not yet a practical default—and definitely not an accuracy claim.

BoT-SORT remains the default. The next question is the one that matters: does McByte actually preserve player identity better through crossings, blocking, and piles?

Project: https://github.com/bgyss/football-tracking

#ComputerVision #SportsAnalytics #MachineLearning

### LinkedIn Post 2: Verifying Identity

A tracking overlay can look convincing and still be wrong about identity.

The next phase of my All-22 tracking project is building the evidence needed to answer a specific question: when a player is occluded, crossed, or absorbed into a pile, does the same anonymous ID come back on the correct player?

That means creating human-reviewed reference tracks and comparing ByteTrack, BoT-SORT, mask-free McByte, and mask-assisted McByte against the same cached detections. I’ll measure IDF1, HOTA, association accuracy, identity switches, fragmentation, and recovery around difficult contact events—not just count tracklets or judge an overlay by eye.

This is anonymous within-shot identity verification, not named-player recognition. Cross-view replay matching is a separate problem and will remain unresolved when the evidence is insufficient.

The standard I want for this project is simple: visible uncertainty is better than a confident-looking false merge.

Project: https://github.com/bgyss/football-tracking

#FootballAnalytics #ComputerVision #MLOps

### LinkedIn Post 3: Fine-Tuning for Football

Generic “person” detection is enough to test a pipeline. It isn’t enough to claim football accuracy.

The current All-22 tracker uses generic RF-DETR Small weights. The next data milestone is a football-specific training set covering wide formations, distant players, officials, sidelines, blocking, tackles, partial visibility, motion blur, and both sideline and end-zone views.

The workflow will keep two kinds of supervision separate:

- detection labels for fine-tuning RF-DETR;
- persistent track labels for evaluating identity through time.

That separation matters. Better boxes can improve tracking inputs, but detector fine-tuning does not by itself prove fewer identity switches.

I’ll split data by game or play so adjacent frames and replay views cannot leak across training and evaluation, use hard-case sampling to decide what to label next, and compare the fine-tuned detector against the generic baseline on held-out games.

The goal isn’t a prettier demo. It’s a pipeline where each improvement has its own measurable evidence.

Project: https://github.com/bgyss/football-tracking

#SportsTech #ComputerVision #MachineLearning

### Reddit Post 1: McByte Integration

#### Title

I integrated mask-assisted McByte into an offline All-22 football tracker

#### Body

I’m building a local pipeline that turns All-22 footage into player detections, anonymous tracks, and trajectories. I recently added Roboflow’s McByte as an opt-in tracker alongside ByteTrack and the current BoT-SORT default.

McByte combines box association with masks initialized by SAM and propagated by Cutie. That makes it an interesting candidate for football footage, where same-team players overlap heavily and box-only association can lose identity through contact.

The integration includes a few controls I wanted before evaluating it:

- explicit mask-on and mask-off modes;
- strict replay of the same complete RF-DETR detection cache across trackers;
- source-frame and timestamp preservation;
- fresh tracker and mask state at each camera cut;
- reporting of requested versus effective mask mode so fallback cannot silently count as a mask-assisted result;
- a reviewed-reference evaluator for HOTA, IDF1, identity switches, fragmentation, and coverage.

I completed a full mask-active run on the 1,424-frame sample. Masks stayed active across both shot-local tracker states with no recorded consecutive mask failures. The tradeoff was substantial: tracking took about 3,771 seconds and peak host memory was about 20,125 MiB on the current Mac.

So the result is “real integration verified,” not “McByte is better.” There is no human-reviewed identity reference yet, so quality remains unevaluated and BoT-SORT remains the default.

Repo: https://github.com/bgyss/football-tracking

I’d be interested in how others annotate identity around piles and partial occlusion, especially when even a human reviewer should be allowed to mark an interval ambiguous.

### Reddit Post 2: Identity Evaluation

#### Title

How I’m planning to verify player identity instead of trusting a tracking overlay

#### Body

The next step in my All-22 tracking project is not another tracker demo. It is a reviewed identity dataset.

For this experiment, “identity” means a persistent anonymous player ID within each camera shot. It does not mean recognizing the player by name, and it does not automatically connect the sideline and end-zone replay views.

The comparison will run ByteTrack, BoT-SORT, mask-free McByte, and mask-assisted McByte over identical cached RF-DETR detections. Human-reviewed tracks will provide the reference for IDF1, HOTA, DetA, AssA, identity switches, fragmentation, and coverage. I also want native-frame event windows around same-team crossings, blocking, piles, and recovery from occlusion, because aggregate metrics can hide the exact failures McByte is meant to address.

A few rules are fixed before seeing the scores:

- missing or unreviewed labels mean `not_evaluated`, not a zero-error result;
- fewer tracklets do not necessarily mean better identity because incorrect merges also reduce track count;
- a mask-disabled fallback cannot count as the mask-assisted variant;
- McByte must reduce total identity switches versus both ByteTrack and BoT-SORT without regressing per-shot IDF1 or HOTA before it becomes a promotion candidate;
- any broader adoption claim needs game-disjoint held-out evaluation.

Repo and evaluation plan: https://github.com/bgyss/football-tracking

For people who have evaluated sports MOT systems: which review protocol worked best for genuinely ambiguous contact sequences?

### Reddit Post 3: Detector Fine-Tuning

#### Title

Next step for my All-22 tracker: football-specific RF-DETR fine-tuning without confusing detection and identity

#### Body

My current pipeline uses generic RF-DETR Small “person” weights. That is useful integration evidence, but wide All-22 footage has its own failure modes: tiny distant players, officials and sideline personnel, camera motion, heavy overlap, partial visibility, tackles, and replay views.

The next training loop will use reviewed football footage with `player`, `official`, and `football` labels. I plan to start with a few hundred diverse keyframes plus contiguous hard sequences, use active learning to prioritize low-confidence detections and crowded events, and keep replay views from the same play in the same split.

The important methodological split is:

- COCO/YOLO boxes train and evaluate the detector;
- persistent MOT-style tracks evaluate identity through time.

A detector can achieve better recall while the tracker still swaps two same-team players. Conversely, a tracker cannot recover a player the detector consistently misses. I want both effects measured separately.

After fine-tuning, I’ll compare the football checkpoint with the generic baseline on held-out games before changing the tracking experiment. That keeps detector gains from being confused with McByte-versus-BoT-SORT association gains.

Repo: https://github.com/bgyss/football-tracking

I’d welcome suggestions on the highest-value All-22 hard negatives or augmentations to include without making jersey details unreadable.

### X Post 1: McByte Integration

McByte is now an opt-in tracker in my All-22 pipeline. A full 1,424-frame mask-active run completed with SAM + Cutie and no mask fallback—but took ~63 minutes and peaked near 20 GB RAM. Integration verified; identity quality not yet. https://github.com/bgyss/football-tracking

### X Post 2: Identity Verification

A clean tracking overlay isn't identity evidence. Next for my All-22 project: human-reviewed tracks, then ByteTrack vs BoT-SORT vs McByte on identical detections using IDF1, HOTA, ID switches, fragmentation, and hard-contact recovery. https://github.com/bgyss/football-tracking

### X Post 3: Fine-Tuning

Generic person weights proved the pipeline, not football accuracy. Next: fine-tune RF-DETR on wide shots, piles, officials, blur, and partial players—then test on held-out games. Detection labels and identity tracks stay separate. https://github.com/bgyss/football-tracking

## Human Review and Calibration Campaign

Drafted September 15, 2026. This campaign is about the review work needed before
claiming that a player identity survives a replay cut. The posts distinguish the
review tooling and evidence contract from the still-missing real-footage accuracy
result.

### LinkedIn Post 1: Human Review Is the Missing Layer

The next phase of my All-22 project is less glamorous than a tracking overlay: making the evidence reviewable.

The pipeline can produce tracklets and candidate cross-shot links. That is not enough to claim that an ID survived a replay cut. A human reviewer now needs to confirm source-frame shot boundaries, semantic field landmarks, timing anchors, player boxes, team labels, and intervals where identity is genuinely ambiguous.

Calibration uses fit landmarks plus independent withheld landmarks. Cross-shot identity uses calibrated field position, reviewed play timing, team compatibility, and visible appearance. When those signals do not agree, the correct output is an abstention.

The full-game review pack preserves original frame numbers and PTS, while keeping detector and Hough proposals separate from reviewed truth. The readiness check remains `not_ready` until reviewed calibration, timing, and identity references exist.

That is slower than showing a confident overlay. It is also the path to a result I can measure.

Project: https://github.com/bgyss/football-tracking

#ComputerVision #SportsAnalytics #MachineLearning

### LinkedIn Post 2: Cross-Shot Identity Starts With Geometry

Image coordinates do not survive a camera cut. A player at pixel `(800, 400)` in a sideline shot tells me almost nothing about the same pixel in an end-zone replay.

The fix I’m testing is a reviewed, time-scoped calibration timeline: map each shot into canonical field coordinates, align the shots with PTS-based play events, then compare player motion only where both views have valid support.

The human review has a deliberate split. Some field landmarks fit the homography; separate landmarks are withheld to test it. Some timing events fit the play-time map; other events are held out to validate it. A mathematically neat fit without independent checks is not identity evidence.

This is the kind of plumbing that rarely makes a demo look better. It determines whether the demo means anything.

Project and review guide: https://github.com/bgyss/football-tracking

#SportsTech #ComputerVision #DataQuality

### LinkedIn Post 3: Ambiguity Belongs in the Dataset

Football footage contains moments where even a careful reviewer cannot prove which player reappears after contact or a partial occlusion.

I’m treating that uncertainty as part of the annotation contract. A reviewer can split a track, mark a frame ignored, leave a team unknown, or abstain from assigning a cross-shot global ID. The evaluator keeps those cases visible instead of turning guesses into false certainty.

The goal is not to force every tracklet into a perfect one-to-one story. The goal is to measure accepted links, missed links, false merges, coverage, and the evidence behind each decision.

If you work with sports video, I’d be interested in how you adjudicate identity around piles, crossings, and camera cuts.

Project: https://github.com/bgyss/football-tracking

#FootballAnalytics #ComputerVision #MLOps

### Reddit Post: How I’m Reviewing All-22 Footage Before Claiming Cross-Shot Identity

#### Title

How I’m reviewing All-22 footage before claiming cross-shot identity

#### Body

I’m building an offline pipeline for player detection, tracking, field calibration, and replay-view identity in All-22 football footage. The next milestone is human review of the evidence, not another tracker demo.

The review has four separate layers:

- **Shot and play structure:** confirm every cut, camera label, source-frame interval, play grouping, and train/validation/test split.
- **Field calibration:** mark semantic yard-line/hash or boundary intersections at each camera-motion keyframe. At least four points fit the homography and at least one independent point is withheld for validation.
- **Play timing:** mark the same visible events in each view using source PTS. Fit correspondences and held-out validation events are kept separate, so the system cannot extrapolate through a replay edit or freeze.
- **Identity reference:** label player boxes, shot-local track IDs, team evidence, visibility, contact confidence, and reviewed cross-shot `global_id` values. Unknown or ambiguous is a valid label.

The review pack keeps detector proposals and Hough line suggestions explicitly unreviewed. The source frame number and PTS stay attached to every annotation, and the source hash is checked before evaluation.

The current real-footage gate is still open: there is no reviewed calibration, timing, or MOT-style identity reference for the full game yet. That means the honest status is `not_ready`, not a fabricated IDF1 or coverage score.

I’d welcome feedback from people who annotate sports video: which contact or replay situations deserve the densest review, and how do you record cases where a human cannot resolve the identity?

Repo and review guide: https://github.com/bgyss/football-tracking

### X Post 1: Review Before the Claim

A tracking overlay is not identity evidence. I’m reviewing All-22 shot boundaries, field landmarks, PTS timing, player boxes, and ambiguous intervals before claiming that an ID survives a replay cut. Unknown is better than a false merge. https://github.com/bgyss/football-tracking

### X Post 2: Calibration Before Cross-Shot Matching

Pixel coordinates do not survive a camera cut. Cross-shot identity needs reviewed field calibration plus PTS-aligned play time, then a conservative match with explicit abstention when the evidence is weak. https://github.com/bgyss/football-tracking

### X Post 3: The Review Pack

I generated a source-hashed review pack for the full All-22. It preserves exact frames and PTS, while keeping detector/Hough proposals unreviewed until a human confirms landmarks, timing, tracks, teams, and cross-shot identity. https://github.com/bgyss/football-tracking

### Media References for This Campaign

Attach these tracked images directly to the posts when a visual is useful:

- [Overview contact sheet](evidence/contact-sheet.jpg): a compact view of the sideline and end-zone sample footage.
- [Cut frames 711–713](evidence/cut-frames-711-713.jpg): the three-frame camera-cut example used to explain why shot-local tracking state must reset.
- [Native sideline frame](evidence/sideline-1s.jpg): useful for showing wide formation scale and small players.
- [Native end-zone frame](evidence/endzone-14s.jpg): useful for showing a second viewpoint and more readable jersey context.

For a local-only review post, attach the generated
`artifacts/full-game-calibration-review-pack/contact-sheet.jpg` and, when rights
permit, a short excerpt from `data/all-22-lions-rams.mp4`. Those files are local
review artifacts and are not public repository URLs; use an authorized uploaded
clip or image when posting publicly.

## Baseline Campaign

### LinkedIn

I'm building a football tracking project with a long-term goal: full reconstruction of football plays from All-22 footage.

The idea is to turn game film into a structured, replayable account of a play: where each player lined up, how they moved, how the ball moved, and how the action unfolded across camera views.

The first step is working locally. The current pipeline combines Roboflow's RF-DETR with BoT-SORT to produce annotated video, anonymous track IDs, and image-space movement trails. It has processed a full 1,424-frame sample containing sideline and end-zone views.

There's still substantial work ahead: football-specific training, keeping player identities consistent through crowded contact, mapping movement into field coordinates, and matching the same players across replay angles. Ball tracking and possession are later pieces of the reconstruction.

This is an early tracking baseline. Full play reconstruction is the destination, and the next step is measuring and improving tracking accuracy against labeled footage.

I'd love to hear from people working in football analytics, coaching, or computer vision: what would make a reconstructed play most useful to you?

#FootballAnalytics #ComputerVision #SportsTech

### X/Twitter

I'm building toward full reconstruction of football plays from All-22 footage: player movement, ball movement, and synchronized views. First step: a local RF-DETR + BoT-SORT tracking baseline. Next: better identities and field calibration. Still early.

### Reddit

#### Title

I'm building an All-22 tracking pipeline, with full football play reconstruction as the long-term goal

#### Body

I'm working on a project to turn All-22 football footage into structured play data. The long-term goal is full reconstruction of football plays: player positions and movement, ball movement and possession, and synchronized sideline/end-zone views that describe the same play.

The current version is an offline Python pipeline using Roboflow's RF-DETR for detection and BoT-SORT for tracking. It produces annotated video, anonymous track IDs, image-space trajectories, and CSV/Parquet exports. I've run it locally on a 23.76-second, 1,424-frame sample with a sideline view followed by an apparent end-zone replay.

A few details about where it stands:

- The sample run uses generic pretrained RF-DETR Small weights. Football-specific fine-tuning and evaluation against labeled player identities are still ahead.
- Tracking resets at camera cuts. Matching players across the two views remains unresolved, and an anonymous track ID isn't yet a guarantee of one player's identity throughout the play.
- The sample trajectories are in image coordinates. Mapping them into yards needs valid field calibration, including handling camera pan and zoom.
- Ball tracking and possession are planned extensions. Full play reconstruction is the long-term goal.

The next work is building the annotation/training loop, improving identities through overlaps and blocking, and aligning the camera views in field coordinates and play time. I want uncertainty to stay visible when the footage doesn't support a reliable match.

For people who work with All-22 or sports tracking: where would you focus first? Which failure cases or outputs would matter most for actual film study?

## Editorial Notes and Sources

These notes support the drafts and are not part of the posts.

- Current capability and sample results: [project README](../README.md), checked against the local `artifacts/verified-botsort/` manifest, metrics, review, and calibration outputs.
- Remaining calibration and identity work: [accuracy, calibration, and replay guide](accuracy-calibration-replay.md).
- Architecture and planned ball/possession work: [system design](system-design.md). Its design-stage status is superseded by the README's implementation status.
- Accuracy claims require the labeled evaluation described in the [evaluation plan](evaluation-plan.md). Processing the sample does not establish tracking accuracy or complete reconstruction.
- McByte implementation status, runtime, memory, and open quality gate: [McByte evaluation evidence](evidence/mcbyte-evaluation.md).
- McByte comparison rules and identity metrics: [McByte evaluation plan](mcbyte-evaluation-plan.md).
- Fine-tuning data and review workflow: [local annotation and data generation](annotation-and-data-generation.md).
- Repository URL: https://github.com/bgyss/football-tracking, derived from the configured `origin` remote.
- Each standalone X post fits the standard 280-character limit. See [X's character-counting documentation](https://docs.x.com/fundamentals/counting-characters).
