# Football Tracking: Social Posts

Drafted September 10, 2026. Each platform section contains ready-to-copy post text. The long-term goal is full reconstruction of football plays from All-22 footage; the current milestone is a local player-detection and tracking baseline.

## LinkedIn

I'm building a football tracking project with a long-term goal: full reconstruction of football plays from All-22 footage.

The idea is to turn game film into a structured, replayable account of a play: where each player lined up, how they moved, how the ball moved, and how the action unfolded across camera views.

The first step is working locally. The current pipeline combines Roboflow's RF-DETR with BoT-SORT to produce annotated video, anonymous track IDs, and image-space movement trails. It has processed a full 1,424-frame sample containing sideline and end-zone views.

There's still substantial work ahead: football-specific training, keeping player identities consistent through crowded contact, mapping movement into field coordinates, and matching the same players across replay angles. Ball tracking and possession are later pieces of the reconstruction.

This is an early tracking baseline. Full play reconstruction is the destination, and the next step is measuring and improving tracking accuracy against labeled footage.

I'd love to hear from people working in football analytics, coaching, or computer vision: what would make a reconstructed play most useful to you?

#FootballAnalytics #ComputerVision #SportsTech

## X/Twitter

I'm building toward full reconstruction of football plays from All-22 footage: player movement, ball movement, and synchronized views. First step: a local RF-DETR + BoT-SORT tracking baseline. Next: better identities and field calibration. Still early.

## Reddit

### Title

I'm building an All-22 tracking pipeline, with full football play reconstruction as the long-term goal

### Body

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
- No public repository URL is configured in this checkout, so the drafts omit a project link.
- The X post uses plain ASCII and fits the standard 280-character limit. See [X's character-counting documentation](https://docs.x.com/fundamentals/counting-characters).
