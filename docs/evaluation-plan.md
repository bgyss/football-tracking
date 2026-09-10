# Tracking evaluation plan

These are proposed engineering acceptance targets, not achieved results. The supplied clip has been inspected but no model has been run. Evaluate the first version on stable player IDs, teams, and trajectories.

## Ground truth and split policy

Create shot boundaries, player boxes, persistent anonymous identities, team labels, visibility flags, and calibration landmarks. Confirm whether the end-zone shot is a replay and annotate its temporal alignment before judging cross-view matches. Permit `unknown` identity and ignore regions when human annotators cannot resolve players; do not silently drop difficult visible players.

For development, inspect the complete 23.757-second clip. Annotate detection keyframes across both shots and dense temporal windows around the snap, crossings, blocking, and tackle. For tracking scores, densely annotate both complete shots at a declared common evaluation cadence (initially approximately 10 Hz, using nearest source timestamps) and inspect transitions at native cadence to catch brief swaps. Fix this sample list before comparing trackers and do not score unannotated frames as empty ground truth.

All variants use the same reference, ignore policy, and temporal sampling. Boxes follow one documented extent policy for partial occlusion/truncation. Fully hidden players have no measurable visible box; recovery after occlusion is evaluated separately. Reference ID ambiguity is adjudicated before reporting results, and excluded intervals are counted explicitly.

Scores on this single play are development evidence. A release evaluation needs at least ten additional plays from at least three games with different formations, camera movement, and uniform pairs; this is an initial breadth target, not a statistically representative NFL sample. Keep all frames and replay views of a play in one split. Training/development and held-out games must be disjoint. Freeze models, thresholds, and evaluation rules before the held-out run.

## Required comparisons

| Variant | Controlled question |
| --- | --- |
| RF-DETR + ByteTrack | How well does a simple local tracker perform? |
| Same detections + BoT-SORT | Does the overall BoT-SORT tracker improve on ByteTrack here? |
| Same BoT-SORT settings, camera compensation on vs off | What improvement is attributable to camera compensation? |
| BoT-SORT + team and local identity resolver | Do football constraints reduce swaps and fragmentation? |
| Same resolver + selective Astra | Does semantic assistance improve identity quality at acceptable cost? |

Cache RF-DETR output so tracker differences are not confounded by detector settings. Separately test detector resolution, tiling, and model size only if recall is inadequate. Compare on equal reference coverage, and disclose processing modes that use future frames. An optional SAM or sports-embedding experiment must beat the simpler path on these same measurements before adoption.

## Proposed gates

| Area | Initial acceptance target | Measurement rule |
| --- | --- | --- |
| Decode/export | All 1,424 source frames accounted for, monotonic timestamps, duration within one source-frame interval | Verify actual decoded count against metadata; preserve frame mapping and audio sync |
| Cuts | Correct boundary at frame 712; no raw track state or pixel trail carried across it | Check boundary frames and output IDs |
| Detection | Visible-player precision and recall each at least 0.95 at IoU 0.5 | Report separately for each view and difficult occlusion strata; no high-score-only filtering of denominators |
| Within-shot identity | IDF1 at least 0.90; at most one ID switch per shot on the sample | Standard identity evaluation at the declared cadence, plus native-frame review around contacts |
| Tracking detail | Report HOTA, DetA, AssA, fragmentation, and lost/recovered tracks | Use a standard pinned evaluator; do not substitute detector mAP for identity metrics |
| Team assignment | At least 0.98 accuracy on assigned visible-player observations, at least 0.95 assignment coverage | Report unknowns and confusion with officials separately |
| Cross-view identity | Zero false merges in the sample; at least 0.80 coverage of human-resolvable shared players | Precision and coverage reported together; unresolved matches remain explicit |
| Calibration | Median withheld-landmark field error at most 1 yard; 95th percentile at most 2 yards | Evaluate by shot and camera-motion interval on landmarks not used for fitting |
| Player positions | Median ground-contact position error at most 1.5 yards, with valid output on at least 0.90 of visible, ground-contact-evaluable observations | Use independent manual foot/contact labels and validated reference mapping; also report errors and nulls in difficult poses |
| Reliability | Valid local results despite disabled/unavailable Astra; resumable from cached detections | Exercise unavailable backend, empty detections, invalid calibration, and budget exhaustion |
| Runtime/cost | Complete instrumented report on the actual execution hardware | Report total and per-stage wall time, throughput, peak memory, requests, usage, and spend; no preset real-time claim |

These thresholds are proposed product goals to review after the first measured baseline. Report failure openly rather than lowering a gate after seeing the held-out result. Aggregate success cannot conceal a failing view or contact-heavy segment.

Use HOTA/IDF1 and identity-switch definitions from a standard evaluator such as [TrackEval](https://github.com/JonathonLuiten/TrackEval/blob/12c8791b303e0a0b50f753af204249e622d0281a/Readme.md); record the selected configuration. For jersey recognition later, report accuracy conditioned on readability and coverage over all observations separately. For possession later, evaluate temporal events and an explicit unknown state against human annotations.

## Correctness tests that matter

Implementation tests should exercise a color-conversion fixture; round-trip model/crop/full-frame coordinates; an empty-detection gap; tracker expiry at 59.94 fps; camera-cut reset; crossing players with conflicting team evidence; a same-shot duplicate identity; calibration degeneracy and drift; replay duplication; and API refusal, timeout, invalid referenced IDs, and budget limits.

These verify plumbing and invariants. They do not prove player-tracking quality on the real clip. The real acceptance artifacts must include the annotated video, field view, reference labels, evaluation configuration, metrics, and failure examples.

## Result reporting

Report **automatic**, **automatic plus Astra**, and **human-corrected** results separately. Show representative crossings and pile recoveries rather than only easy pre-snap frames. Preserve every manual change with its author/source and affected interval. State how many IDs, team labels, and field positions remain unknown.

Before accepting the system, watch the entire output in both shots and inspect unresolved cases. Confirm that the end-zone crop does not fabricate off-screen players, field trajectories do not jump with the camera, and replay footage does not add duplicate distance or events. Include held-out performance separately from this sample's development results.
