# Cross-Shot Identity Automation Implementation Plan

> **Status:** Implemented in the current worktree and manually smoke-checked on generated local video. Full-game identity quality and CVAT server operation remain unmeasured; generated artifacts stay unreviewed until a human records approval.

**Goal:** Implement the prioritized automation path in [the 2026-09-22 research note](../../evidence/cross-shot-identity-automation-research-2026-09-22.md) while preserving source-frame, PTS, calibration, timing, and review provenance.

**Architecture:** Add a dependency-free CVAT XML interchange layer driven by an explicit task-frame map. Reuse existing tracking, PTS alignment, calibration timeline, and `identity-links.json` evidence to produce correction proposals and a source-linked review queue. Keep learned OCR/ReID and semantic field detections optional, provenance-carrying proposal inputs; automatic evidence never becomes reviewed identity truth.

**Tech Stack:** Python 3.11+, standard-library XML/JSON/CSV/ZIP, existing OpenCV/NumPy video stack, optional local Tesseract executable when explicitly requested.

**Spec:** `docs/evidence/cross-shot-identity-automation-research-2026-09-22.md`, `docs/annotation-schema.md`, and `docs/human-review-all22.md`.

## Global Constraints

- Keep frame numbers zero-based in source coordinates and preserve the integer source PTS plus `time_base` for every imported/exported label.
- Keep every generated manifest, cut, event, box, landmark, contact, team, jersey, embedding, and identity proposal explicitly unreviewed until a reviewer records approval metadata.
- Keep the canonical field orientation `x=0..120`, `y=0..53⅓` yards and existing withheld-landmark gates (median ≤1 yard, p95 ≤2 yards).
- Preserve the current conservative assignment, explicit unmatched outcome, component guards, and `ready_for_evaluation` prerequisites.
- Add no required CVAT SDK, model weights, cloud calls, or new heavyweight core dependencies.
- Keep source paths and video/model outputs out of committed annotations; hash the source and map task-local frames through a sidecar.

## Review Focus

- A cropped or sampled CVAT task can map a correct task frame to the wrong source frame or PTS; validate every frame-map entry and coordinate transform.
- CVAT `outside` shapes can accidentally become visible empty annotations; preserve visibility intervals without fabricating objects.
- Multiple shapes can share a local CVAT track ID across source shots; scope IDs by shot and reject ambiguous mappings.
- Automatic identity candidates can be mistaken for labels; candidate artifacts must remain unreviewed and must not feed the readiness gate as reference truth.
- Missing PTS, duplicate mappings, non-monotonic timing, invalid XML, or source hash mismatch must fail import with actionable errors.

## File Map

- `src/football_tracking/cvat.py` — task-frame maps, CVAT box/point XML read/write, source-coordinate conversion, and proposal/reference status handling.
- `scripts/export_cvat.py` — export source-addressed observation tracks and frame-map sidecar for one shot or bounded task.
- `scripts/import_cvat.py` — import corrected CVAT XML through the frame map into a versioned unreviewed annotation manifest; reviewed promotion requires explicit reviewer metadata.
- `src/football_tracking/identity_review.py` — turn identity candidate evidence and observations into prioritized source-frame review items with stable IDs.
- `scripts/build_identity_review_queue.py` — command-line review queue export with optional local contact sheet.
- `src/football_tracking/replay.py`, `identity_resolution.py`, and `cli.py` — extend candidate evidence with optional, provenance-bearing tracklet cues without changing behavior when cues are absent.
- `scripts/build_play_inventory.py` and `scripts/build_identity_review_pack.py` — retain and expose exact source PTS mappings and unreviewed event/landmark proposals in generated review inputs.
- `docs/annotation-and-data-generation.md`, `docs/annotation-schema.md`, and `docs/human-review-all22.md` — describe the supported local loop and its review state transitions.

## Tasks

### Task 1: Define source-addressed CVAT contracts

Implement immutable task-frame map records with `task_frame`, `source_frame`, `source_pts`, source hash, source time base, source dimensions, task dimensions, and optional crop rectangle. Reject duplicate task/source frames, missing or negative PTS, non-monotonic source mappings, invalid crop bounds, and source-hash mismatch. Implement CVAT XML 1.1 video track serialization and parsing using only the standard library, including box and point shapes, `outside`, `occluded`, `keyframe`, label, track ID, and shape attributes. Preserve unknown XML attributes when practical. **Status: complete.**

Manual verification completed: generated-video XML round-trip preserved exact source frame/PTS and recovered original box coordinates after a crop and resize; separate point tracks imported correctly.

### Task 2: Export tracker proposals to CVAT

Read `observations.csv`, select one declared shot and source-frame interval, map source frames to task-local frames, and write standalone CVAT XML, a repository bundle, and an explicit frame-map sidecar. Export proposal labels/attributes for `tracklet_id`, team suggestion, detection score, source PTS, and provenance. Emit box and bottom-center contact point proposals; when a review pack is supplied, also emit field-intersection point proposals. Emit every observed box as a keyframe and preserve the crop transform. **Status: complete.**

Manual verification completed: standalone XML, repository bundle, and sidecar were inspected on a generated video; source-space box and point transforms round-tripped exactly.

### Task 3: Import CVAT corrections safely

Import CVAT XML and its frame map into an annotation-manifest template, translating task-local frame IDs to original source frame and PTS, and transformed boxes/points to source pixels. Preserve `outside` and `occluded` state, shot-local track IDs, team attributes, landmarks, contact points, and task/source provenance. Incomplete point semantics stay in `point_proposals` and block promotion. Default all imported records and the manifest to `reviewed: false`; add an explicit promotion option that requires reviewer, revision, timestamp, and confidence fields and validates the completed manifest through `load_annotation_manifest`. Allow reviewed source-pixel contacts to project through a matching validated calibration timeline when building the evaluator reference. Require independent second-review metadata for reviewed cross-shot decisions. **Status: complete.**

Manual verification completed: generated-video box/contact/landmark tracks became a reviewed manifest; a validated timeline projected a reviewed pixel contact, and a cross-shot ID without an independent second review was rejected.

### Task 4: Build a cross-shot identity review queue

Read `identity-links.json` and `observations.csv`; rank accepted, ambiguous, rejected, and unmatched cases by assignment margin, score competition, rejection reason, overlap length, and missing evidence. Select observations nearest the candidate overlap midpoint in common play time within 100 ms and include source frame, source PTS, time base, field coordinates, team, calibration ID, confidence values, and the full evidence record. If observations are unavailable, preserve the candidate with a missing-frame reason. Write deterministic JSON and, when requested, a side-by-side contact sheet from the original source; queue status is always `unreviewed`. **Status: complete.**

Manual verification completed: a generated-video queue retained exact source PTS, ranked an ambiguous candidate, and wrote a contact sheet from matching frames.

### Task 5: Feed optional multi-cue proposals into candidate evidence

Define a source-hashed and base-analysis-bound tracklet-cue JSON contract for per-frame jersey OCR readings and appearance embeddings, with crop/source frame/PTS, box, model provenance, crop quality, confidence, and explicit proposal/review state. Aggregate repeated jersey readings and embedding similarity across quality-filtered crops. Add cue values, quality summaries, and missingness to each candidate record. Only reviewed reliable conflicting jersey values may be a hard rejection; unreviewed OCR/ReID outputs may rank review candidates but cannot create accepted identity links by themselves. **Status: complete.**

Manual verification completed: Tesseract proposed the synthetic jersey number `24` on three source frames; cue loading matched tracklet/frame/PTS/box and analysis hash; a reviewed conflicting number vetoed a link, while proposal cues remained review-ranking evidence. No NFL OCR accuracy claim was measured.

### Task 6: Extend cut, event, calibration, and contact proposal exports

Ensure shot inventory and review packs carry exact source PTS values; mark the end-of-file interval as having no exclusive-boundary PTS. Add overlapping adjacent-shot play-window proposals and bounded unreviewed motion-burst/camera-motion keyframes without auto-confirming events or building a reviewed time map. Add Hough field-line intersection and box-bottom contact point proposals for CVAT review. Continue to use reviewed landmarks and held-out events as the sole calibration/timing acceptance evidence. **Status: complete.**

Manual verification completed: tiny-video inventory windows, review-pack points, motion-burst candidates, and camera keyframes retained exact PTS and `reviewed: false`. No full-game proposal quality was measured.

### Task 7: Document commands, readiness limits, and local workflow

Document local CVAT export/import commands, task-map rules for contiguous clips and crops, review-promotion requirements, independent identity review, identity queue fields, optional cue inputs, and calibration/timing limitations. State that completion of automation does not make an unreviewed pack ready for evaluation. **Status: complete.**

Manual verification completed: documented command interfaces load, relative Markdown links resolve, and generated manifests validate locally. The supplied full-game source, local CVAT server, and readiness gate were not run.
