# Cross-Shot Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make cross-shot player identity measurable and then resolvable, so that a tracklet in the end-zone replay can be joined to the same player's tracklet in the sideline shot under a predeclared zero-false-merge gate.

**Architecture:** Add a measurement layer first (`evaluation.py` gains a cross-shot identity scorer and the reviewed-reference schema gains an optional global-identity map), then a resolution layer (`replay.py` aligns shots on a reviewed snap anchor and produces candidate scores) that feeds the *already implemented and already tested* `match_tracklets` → `stable_anonymous_ids` path in `identity.py`. The CLI wires them together behind an explicit `--play-alignment` flag; with no flag, behavior is byte-identical to today except for a new `identity-links.json` evidence file.

**Tech Stack:** Python 3.11+, numpy, scipy (`linear_sum_assignment`, already an optional-import path), pytest, `uv`.

---

## Findings: what the artifacts actually show

Investigated on 2026-09-14 against `artifacts/mcbyte-evaluation/` at commit `d5ca65d`.

**McByte is not the cause.** `artifacts/mcbyte-evaluation/mcbyte-mask-on-unbounded/metrics.json` records `masks_active: true`, `fallback_reason: null`, and `consecutive_mask_failures: 0` for both `shot-0` and `shot-1`. The adapter reset cleanly at the cut and produced 18,343 observations over 52 tracklets with `duplicate_track_ids_within_frame: 0`. The McByte integration is doing what it was built to do.

**Cross-shot identity is unimplemented for every tracker, by design.** `src/football_tracking/cli.py:340` calls:

```python
identity_map = stable_anonymous_ids(tracklet_ids, [])
```

The second argument — the `IdentityLink` list — is a hardcoded empty list. No caller ever produces links, and `match_tracklets` (`identity.py:139`) is exercised only by `tests/test_identity.py`. The observable consequence, confirmed across three runs:

| Run | Tracklets | Unique player IDs | shot-0 / shot-1 | Player IDs spanning >1 tracklet |
| --- | --- | --- | --- | --- |
| `mcbyte-mask-on-unbounded` | 52 | 52 | 24 / 28 | 0 |
| `mcbyte-mask-off` | 52 | 52 | 24 / 28 | 0 |
| `verified-botsort` | 59 | 59 | 27 / 32 | 0 |

`identities.json` is a 1:1 tracklet→ID map in every case. `review.json` reports this honestly as `{"status": "unresolved_cross_view", "cross_view_identity_resolved": false}`. This matches the deliberate design recorded in `docs/accuracy-calibration-replay.md:96`.

**Two different failures look identical in the current artifacts.** A player ID that "doesn't lock in" can mean (a) the tracklet was correctly tracked through its shot but never joined to its counterpart across the cut, or (b) the tracker fragmented the player *within* a shot, producing several tracklets and therefore several IDs. Today we cannot distinguish these, because `tracking-evaluation.json` says `{"status": "not_evaluated"}` — no reviewed reference has ever been supplied. Shot-0 having 24 tracklets for ~22 players is consistent with either reading.

**This plan therefore measures before it fixes.** Tasks 1–2 make both failure modes visible; Tasks 3–5 implement the join; Task 6 records the result.

## Milestone: M4 — Cross-shot replay identity

**Definition of done.** `player_id` is stable across the frame-712 cut for players a human reviewer can resolve in both views, under the gate predeclared in `docs/evaluation-plan.md:39` and `docs/accuracy-calibration-replay.md:110`:

> Zero false merges in the sample; at least 0.80 coverage of human-resolvable shared players. Precision and coverage reported together; unresolved matches remain explicit.

**Exit states, reported independently** (following the convention in `docs/goals/mcbyte-integration.md`):

- **Plumbing ready.** Alignment loader, candidate scorer, CLI wiring, `identity-links.json`, and the cross-shot evaluator are implemented; full suite passes; default runs are unchanged.
- **Measured baseline.** A reviewed reference with a `cross_shot_identity` map exists; the current no-link baseline is scored and reported (expected: coverage 0.0, false merges 0).
- **Resolution verified.** With reviewed snap anchors and shot-specific calibration landmarks, the resolver produces accepted links and the gate is scored — pass or fail reported as measured.

**Blocked prerequisites** — these are inputs, not code, and each blocks only the state it names:

| Prerequisite | Blocks | Why |
| --- | --- | --- |
| Reviewed MOT-style reference with global identity map | Measured baseline, Resolution verified | `evaluate_tracking` refuses unreviewed data by design (`evaluation.py:45`) |
| Reviewed snap-frame anchors for both shots | Resolution verified | Play-time alignment has no other anchor |
| Shot-specific calibration landmarks for both shots | Resolution verified | Field-position agreement is the only view-invariant geometric signal; image coordinates are meaningless across a cut |

Until calibration landmarks exist for both shots, the resolver is **expected and required to abstain** on the sample clip. A measured abstention is a correct outcome for this milestone's plumbing state, not a failure. Do not substitute image-space proximity to manufacture merges.

## Out of scope

Deliberately excluded so this milestone stays one testable subsystem:

- **Jersey-number consensus** as a scoring component. `Observation.jersey_number` is `None`
  everywhere in the pipeline today; there is no recognizer to consense over. The scorer is
  structured so a fourth weighted component can be added without changing its signature.
- **Automatic replay detection.** Whether shot-1 replays shot-0 is a reviewed input here, not
  an inference. `docs/accuracy-calibration-replay.md:102` asks for formation and field-marking
  confirmation; that is its own piece of work.
- **GPT-6 Astra ambiguity arbitration.** The predeclared order (`docs/accuracy-calibration-replay.md:119`)
  puts Astra after the local resolver has a measured ambiguity queue. This plan produces that
  queue — the `insufficient_evidence` links in `identity-links.json` — and stops there.
- **Time-varying calibration.** A single homography per shot is assumed. If withheld-landmark
  error fails the calibration gate, that is a separate milestone.

## Global Constraints

- Default behavior must not change. Without `--play-alignment`, `run` produces the same identity map it produces today.
- BoT-SORT remains the default tracker. This work is tracker-agnostic and must not special-case McByte.
- Never merge on image-coordinate proximity across a cut. Cross-shot evidence is team agreement, calibrated field position at aligned play time, trajectory shape, and jersey consensus only.
- A team-label conflict, missing calibration, or insufficient temporal overlap produces **no candidate at all**, which `match_tracklets` renders as `insufficient_evidence`. Never emit a low-confidence `same`.
- Unreviewed or model-generated references are refused. Any reference lacking `reviewed: true` raises `EvaluationError`.
- `shot_id`, `play_id`, and `player_id` stay separate fields with separate meanings (`docs/accuracy-calibration-replay.md:91`).
- A correct replay merge represents one play observed twice. It must never double-count distance, events, or participation.
- Outputs stay deterministic and sorted. New JSON writers go through `write_metrics_json` in `export.py`.
- No new mandatory dependency. scipy is already core; nothing here may require `trackers`, `rfdetr`, or `trackeval` to import.
- Run every command from the project root with `UV_CACHE_DIR=.uv-cache uv run ...`.
- Do not lower a gate after seeing a result.

---

### Task 1: Cross-shot identity reference schema and evaluator

Measurement comes first: the current state cannot be called broken until it is scored.

**Files:**
- Modify: `src/football_tracking/evaluation.py` (add after `load_reviewed_mot_reference`, which ends at line 80)
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `ReferenceFrame`, `ReferenceObject`, `EvaluationError`, `_iou` — all already in `evaluation.py`.
- Produces:
  - `load_cross_shot_identity(path: str | Path) -> dict[str, dict[str, str]] | None`
  - `dominant_reference_ids(rows: Sequence[TrackObservation], frames: Mapping[int, ReferenceFrame], iou_threshold: float = 0.5) -> dict[str, str]`
  - `evaluate_cross_shot_identity(identity_map: Mapping[str, str], predictions: Sequence[TrackObservation], reference: Mapping[str, Mapping[int, ReferenceFrame]], cross_shot_identity: Mapping[str, Mapping[str, str]] | None, *, iou_threshold: float = 0.5) -> dict[str, Any]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluation.py`:

```python
def test_cross_shot_identity_loader_requires_reviewed_and_returns_none_when_absent(tmp_path) -> None:
    from football_tracking.evaluation import load_cross_shot_identity

    without = tmp_path / "without.json"
    without.write_text(json.dumps({"reviewed": True, "sequences": {"shot-0": {"frames": {}}}}), encoding="utf-8")
    assert load_cross_shot_identity(without) is None

    with_map = tmp_path / "with.json"
    with_map.write_text(json.dumps({
        "reviewed": True,
        "sequences": {"shot-0": {"frames": {}}},
        "cross_shot_identity": {"shot-0": {"p1": "PLAYER-A"}, "shot-1": {"p9": "PLAYER-A"}},
    }), encoding="utf-8")
    assert load_cross_shot_identity(with_map) == {"shot-0": {"p1": "PLAYER-A"}, "shot-1": {"p9": "PLAYER-A"}}

    unreviewed = tmp_path / "unreviewed.json"
    unreviewed.write_text(json.dumps({"cross_shot_identity": {"shot-0": {"p1": "PLAYER-A"}}}), encoding="utf-8")
    with pytest.raises(EvaluationError):
        load_cross_shot_identity(unreviewed)


def _two_shot_reference(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({
        "reviewed": True,
        "sequences": {
            "shot-0": {"frames": {
                "0": {"labeled": True, "objects": [
                    {"id": "p1", "bbox_xyxy": [0, 0, 10, 10]},
                    {"id": "p2", "bbox_xyxy": [50, 50, 60, 60]},
                ]},
            }},
            "shot-1": {"frames": {
                "100": {"labeled": True, "objects": [
                    {"id": "q1", "bbox_xyxy": [0, 0, 10, 10]},
                    {"id": "q2", "bbox_xyxy": [50, 50, 60, 60]},
                ]},
            }},
        },
        "cross_shot_identity": {
            "shot-0": {"p1": "PLAYER-A", "p2": "PLAYER-B"},
            "shot-1": {"q1": "PLAYER-A", "q2": "PLAYER-B"},
        },
    }), encoding="utf-8")
    return path


def test_cross_shot_evaluation_reports_zero_coverage_for_unlinked_identities(tmp_path) -> None:
    from football_tracking.evaluation import evaluate_cross_shot_identity, load_cross_shot_identity

    path = _two_shot_reference(tmp_path)
    reference = load_reviewed_mot_reference(path)
    links = load_cross_shot_identity(path)
    predictions = [
        track("shot-0:t1", 0, (0.0, 0.0, 10.0, 10.0)),
        track("shot-0:t2", 0, (50.0, 50.0, 60.0, 60.0)),
        track("shot-1:t1", 100, (0.0, 0.0, 10.0, 10.0)),
        track("shot-1:t2", 100, (50.0, 50.0, 60.0, 60.0)),
    ]
    identity_map = {"shot-0:t1": "P01", "shot-0:t2": "P02", "shot-1:t1": "P03", "shot-1:t2": "P04"}

    report = evaluate_cross_shot_identity(identity_map, predictions, reference, links)

    assert report["status"] == "evaluated"
    assert report["resolvable_pairs"] == 2
    assert report["merged_pairs"] == 0
    assert report["true_merges"] == 0
    assert report["false_merges"] == 0
    assert report["coverage"] == pytest.approx(0.0)
    assert report["precision"] is None


def test_cross_shot_evaluation_separates_true_and_false_merges(tmp_path) -> None:
    from football_tracking.evaluation import evaluate_cross_shot_identity, load_cross_shot_identity

    path = _two_shot_reference(tmp_path)
    reference = load_reviewed_mot_reference(path)
    links = load_cross_shot_identity(path)
    predictions = [
        track("shot-0:t1", 0, (0.0, 0.0, 10.0, 10.0)),
        track("shot-0:t2", 0, (50.0, 50.0, 60.0, 60.0)),
        track("shot-1:t1", 100, (0.0, 0.0, 10.0, 10.0)),
        track("shot-1:t2", 100, (50.0, 50.0, 60.0, 60.0)),
    ]
    # t1 pair is correct (both PLAYER-A); t2 of shot-0 is wrongly merged with t1 of shot-1.
    identity_map = {"shot-0:t1": "P01", "shot-1:t1": "P01", "shot-0:t2": "P02", "shot-1:t2": "P02"}
    good = evaluate_cross_shot_identity(identity_map, predictions, reference, links)
    assert good["true_merges"] == 2
    assert good["false_merges"] == 0
    assert good["coverage"] == pytest.approx(1.0)
    assert good["precision"] == pytest.approx(1.0)

    wrong = {"shot-0:t2": "P01", "shot-1:t1": "P01", "shot-0:t1": "P02", "shot-1:t2": "P02"}
    bad = evaluate_cross_shot_identity(wrong, predictions, reference, links)
    assert bad["false_merges"] == 2
    assert bad["true_merges"] == 0
    assert bad["precision"] == pytest.approx(0.0)


def test_cross_shot_evaluation_is_not_evaluated_without_identity_map(tmp_path) -> None:
    from football_tracking.evaluation import evaluate_cross_shot_identity

    path = _two_shot_reference(tmp_path)
    reference = load_reviewed_mot_reference(path)
    report = evaluate_cross_shot_identity({}, [], reference, None)
    assert report["status"] == "not_evaluated"
    assert "cross_shot_identity" in report["reason"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_evaluation.py -k cross_shot -v
```

Expected: FAIL with `ImportError: cannot import name 'load_cross_shot_identity'`.

- [ ] **Step 3: Implement the loader, attribution, and evaluator**

Insert into `src/football_tracking/evaluation.py` immediately after `load_reviewed_mot_reference` (i.e. after line 80):

```python
def load_cross_shot_identity(path: str | Path) -> dict[str, dict[str, str]] | None:
    """Load the optional reviewed map from sequence-local object ids to global player ids.

    Returns None when the reviewed reference simply does not carry the map, so a
    caller can report ``not_evaluated`` rather than inventing a cross-shot score.
    """

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"unable to read reference: {error}") from error
    if not isinstance(value, dict) or value.get("reviewed") is not True:
        raise EvaluationError("reference must be explicitly marked reviewed: true")
    raw = value.get("cross_shot_identity")
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise EvaluationError("cross_shot_identity must be a non-empty object when present")
    result: dict[str, dict[str, str]] = {}
    for shot_id, mapping in raw.items():
        if not isinstance(mapping, dict) or not mapping:
            raise EvaluationError(f"cross_shot_identity[{shot_id!r}] must be a non-empty object")
        result[str(shot_id)] = {str(key): str(name) for key, name in mapping.items()}
    return result


def dominant_reference_ids(
    rows: Sequence[TrackObservation],
    frames: Mapping[int, ReferenceFrame],
    iou_threshold: float = 0.5,
) -> dict[str, str]:
    """Attribute each tracklet to the reference object it overlaps most often."""

    votes: dict[str, dict[str, int]] = {}
    for row in rows:
        frame = frames.get(row.frame_index)
        if frame is None or not frame.labeled or frame.ignore:
            continue
        best_id, best_iou = None, iou_threshold
        for reference_object in frame.objects:
            score = _iou(np.asarray(row.bbox_xyxy_px, dtype=float), np.asarray(reference_object.bbox_xyxy, dtype=float))
            if score >= best_iou:
                best_id, best_iou = reference_object.identifier, score
        if best_id is not None:
            votes.setdefault(row.tracklet_id, {}).setdefault(best_id, 0)
            votes[row.tracklet_id][best_id] += 1
    # Ties break on the lexicographically smallest id so the result is deterministic.
    return {
        tracklet_id: min(sorted(counts), key=lambda key: (-counts[key], key))
        for tracklet_id, counts in votes.items()
    }


def evaluate_cross_shot_identity(
    identity_map: Mapping[str, str],
    predictions: Sequence[TrackObservation],
    reference: Mapping[str, Mapping[int, ReferenceFrame]],
    cross_shot_identity: Mapping[str, Mapping[str, str]] | None,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score cross-shot merges as precision and coverage, never as a single number."""

    if not cross_shot_identity:
        return {"status": "not_evaluated", "reason": "reviewed reference carries no cross_shot_identity map"}
    by_shot: dict[str, list[TrackObservation]] = {}
    for row in predictions:
        by_shot.setdefault(row.tracklet_id.split(":", 1)[0], []).append(row)
    truth: dict[str, str] = {}
    for shot_id, frames in reference.items():
        shot_map = cross_shot_identity.get(shot_id, {})
        for tracklet_id, reference_id in dominant_reference_ids(by_shot.get(shot_id, []), frames, iou_threshold).items():
            global_id = shot_map.get(reference_id)
            if global_id is not None:
                truth[tracklet_id] = global_id
    attributed = sorted(truth)
    resolvable = 0
    merged = 0
    true_merges = 0
    false_merges = 0
    false_merge_examples: list[dict[str, str]] = []
    missed_examples: list[dict[str, str]] = []
    for index, left in enumerate(attributed):
        for right in attributed[index + 1 :]:
            if left.split(":", 1)[0] == right.split(":", 1)[0]:
                continue
            same_player = truth[left] == truth[right]
            same_id = identity_map.get(left) is not None and identity_map.get(left) == identity_map.get(right)
            if same_player:
                resolvable += 1
            if same_id:
                merged += 1
            if same_id and same_player:
                true_merges += 1
            elif same_id and not same_player:
                false_merges += 1
                if len(false_merge_examples) < 20:
                    false_merge_examples.append({"left": left, "right": right, "left_player": truth[left], "right_player": truth[right]})
            elif same_player and not same_id:
                if len(missed_examples) < 20:
                    missed_examples.append({"left": left, "right": right, "player": truth[left]})
    return {
        "status": "evaluated",
        "iou_threshold": iou_threshold,
        "attributed_tracklets": len(truth),
        "resolvable_pairs": resolvable,
        "merged_pairs": merged,
        "true_merges": true_merges,
        "false_merges": false_merges,
        "coverage": true_merges / resolvable if resolvable else 0.0,
        "precision": (true_merges / merged) if merged else None,
        "gate": {
            "false_merges_zero": false_merges == 0,
            "coverage_at_least_0_80": (true_merges / resolvable if resolvable else 0.0) >= 0.80,
        },
        "false_merge_examples": false_merge_examples,
        "missed_pair_examples": missed_examples,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_evaluation.py -v
```

Expected: PASS, including the pre-existing tests.

- [ ] **Step 5: Commit**

```bash
git add src/football_tracking/evaluation.py tests/test_evaluation.py
git commit -m "feat: score cross-shot identity as precision and coverage"
```

---

### Task 2: Emit identity link evidence from every run

Makes the current state visible in artifacts without changing it, so "IDs did not lock in" becomes a readable line in a file rather than an inference from `identities.json`.

**Files:**
- Modify: `src/football_tracking/cli.py:340` (identity map construction) and `cli.py:366-383` (the export stage)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `stable_anonymous_ids`, `IdentityLink` from `identity.py`; `write_metrics_json` from `export.py`.
- Produces: `artifacts/<run>/identity-links.json` with keys `status`, `shot_count`, `links`, `player_id_count`, `tracklet_count`, `cross_shot_player_ids`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_run_emits_identity_links_evidence_and_defaults_to_unlinked(tmp_path) -> None:
    from football_tracking.cli import main

    output = tmp_path / "run"
    exit_code = main([
        "run", "--input", "data/all-22-lions-rams-sample.mp4", "--output", str(output),
        "--detector", "synthetic", "--tracker", "iou", "--manual-cut", "712",
    ])
    assert exit_code == 0

    report = json.loads((output / "identity-links.json").read_text(encoding="utf-8"))
    assert report["status"] == "not_attempted"
    assert report["reason"] == "--play-alignment was not provided"
    assert report["links"] == []
    assert report["cross_shot_player_ids"] == 0
    assert report["player_id_count"] == report["tracklet_count"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_cli.py::test_run_emits_identity_links_evidence_and_defaults_to_unlinked -v
```

Expected: FAIL with `FileNotFoundError` on `identity-links.json`.

- [ ] **Step 3: Implement the evidence writer**

In `src/football_tracking/cli.py`, replace line 340:

```python
    identity_map = stable_anonymous_ids(tracklet_ids, [])
```

with:

```python
    identity_links: list[IdentityLink] = []
    link_report: dict[str, Any] = {"status": "not_attempted", "reason": "--play-alignment was not provided"}
    identity_map = stable_anonymous_ids(tracklet_ids, identity_links)
    by_player: dict[str, set[str]] = {}
    for tracklet_id, player_id in identity_map.items():
        by_player.setdefault(player_id, set()).add(tracklet_id.split(":", 1)[0])
    link_report.update({
        "shot_count": len(boundaries),
        "tracklet_count": len(tracklet_ids),
        "player_id_count": len(set(identity_map.values())),
        "cross_shot_player_ids": sum(1 for shots in by_player.values() if len(shots) > 1),
        "links": [
            {"left_key": link.left_key, "right_key": link.right_key, "decision": link.decision, "score": link.score}
            for link in identity_links
        ],
    })
```

Add `IdentityLink` to the existing identity import on line 18:

```python
from .identity import IdentityLink, TrackletSummary, resolve_teams, stable_anonymous_ids, team_feature_from_crop
```

Inside the `with timer.stage("export"):` block, immediately after the `write_identities_json(...)` call, add:

```python
        write_metrics_json(destination / "identity-links.json", link_report)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Reproduce the reported symptom as recorded evidence**

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run \
  --input data/all-22-lions-rams-sample.mp4 \
  --output artifacts/cross-shot-baseline \
  --detector synthetic --tracker iou --manual-cut 712
python3 -c "import json;print(json.load(open('artifacts/cross-shot-baseline/identity-links.json')))"
```

Expected: `cross_shot_player_ids` is `0` and `player_id_count` equals `tracklet_count`, confirming the finding above.

- [ ] **Step 6: Commit**

```bash
git add src/football_tracking/cli.py tests/test_cli.py
git commit -m "feat: record identity link evidence in every run"
```

---

### Task 3: Reviewed play-time alignment

**Files:**
- Create: `src/football_tracking/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `class PlayAnchor` with fields `shot_id: str`, `source_frame: int`, `event: str`
  - `class PlayAlignment` with fields `play_id: str`, `anchors: tuple[PlayAnchor, ...]`, methods `shots() -> tuple[str, ...]` and `play_time_s(shot_id: str, frame_index: int, fps: float) -> float | None`
  - `class ReplayAlignmentError(ValueError)`
  - `load_play_alignment(path: str | Path) -> PlayAlignment`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_replay.py`:

```python
from __future__ import annotations

import json

import pytest

from football_tracking.replay import (
    PlayAlignment,
    PlayAnchor,
    ReplayAlignmentError,
    load_play_alignment,
)


def test_play_time_is_anchor_relative_and_shot_scoped() -> None:
    alignment = PlayAlignment("play-1", (PlayAnchor("shot-0", 120, "snap"), PlayAnchor("shot-1", 820, "snap")))

    assert alignment.shots() == ("shot-0", "shot-1")
    assert alignment.play_time_s("shot-0", 120, 60.0) == pytest.approx(0.0)
    assert alignment.play_time_s("shot-1", 820, 60.0) == pytest.approx(0.0)
    assert alignment.play_time_s("shot-0", 150, 60.0) == pytest.approx(0.5)
    assert alignment.play_time_s("shot-1", 850, 60.0) == pytest.approx(0.5)
    assert alignment.play_time_s("shot-2", 10, 60.0) is None


def test_alignment_requires_review_two_shots_and_positive_fps(tmp_path) -> None:
    unreviewed = tmp_path / "unreviewed.json"
    unreviewed.write_text(json.dumps({"play_id": "play-1", "anchors": []}), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError):
        load_play_alignment(unreviewed)

    single = tmp_path / "single.json"
    single.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [{"shot_id": "shot-0", "source_frame": 120, "event": "snap"}],
    }), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError):
        load_play_alignment(single)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 120, "event": "snap"},
            {"shot_id": "shot-0", "source_frame": 130, "event": "snap"},
        ],
    }), encoding="utf-8")
    with pytest.raises(ReplayAlignmentError):
        load_play_alignment(duplicate)

    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 120, "event": "snap"},
            {"shot_id": "shot-1", "source_frame": 820, "event": "snap"},
        ],
    }), encoding="utf-8")
    alignment = load_play_alignment(good)
    assert alignment.play_id == "play-1"
    assert alignment.shots() == ("shot-0", "shot-1")

    with pytest.raises(ValueError):
        alignment.play_time_s("shot-0", 120, 0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_replay.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'football_tracking.replay'`.

- [ ] **Step 3: Implement the alignment module**

Create `src/football_tracking/replay.py`:

```python
"""Reviewed play-time alignment and cross-shot identity candidate scoring.

Shots are tracked independently, so nothing in image coordinates survives a cut.
This module converts media time into a shared play time anchored on a reviewed
event, and scores cross-shot tracklet pairs only from view-invariant evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class ReplayAlignmentError(ValueError):
    """Raised when an alignment cannot support a cross-shot join."""


@dataclass(frozen=True, slots=True)
class PlayAnchor:
    shot_id: str
    source_frame: int
    event: str

    def __post_init__(self) -> None:
        if self.source_frame < 0:
            raise ValueError("anchor source_frame must be non-negative")
        if not self.event:
            raise ValueError("anchor event must be named")


@dataclass(frozen=True, slots=True)
class PlayAlignment:
    play_id: str
    anchors: tuple[PlayAnchor, ...]

    def shots(self) -> tuple[str, ...]:
        return tuple(sorted(anchor.shot_id for anchor in self.anchors))

    def play_time_s(self, shot_id: str, frame_index: int, fps: float) -> float | None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        for anchor in self.anchors:
            if anchor.shot_id == shot_id:
                return (frame_index - anchor.source_frame) / fps
        return None


def load_play_alignment(path: str | Path) -> PlayAlignment:
    """Load reviewed snap anchors. Model-generated alignments are refused."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayAlignmentError(f"unable to read alignment: {error}") from error
    if not isinstance(value, dict) or value.get("reviewed") is not True:
        raise ReplayAlignmentError("alignment must be explicitly marked reviewed: true")
    play_id = str(value.get("play_id") or "")
    if not play_id:
        raise ReplayAlignmentError("alignment must name a play_id")
    raw_anchors = value.get("anchors")
    if not isinstance(raw_anchors, list) or len(raw_anchors) < 2:
        raise ReplayAlignmentError("alignment needs a reviewed anchor for at least two shots")
    anchors: list[PlayAnchor] = []
    seen: set[str] = set()
    for raw in raw_anchors:
        if not isinstance(raw, dict) or "shot_id" not in raw or "source_frame" not in raw:
            raise ReplayAlignmentError(f"invalid anchor: {raw!r}")
        shot_id = str(raw["shot_id"])
        if shot_id in seen:
            raise ReplayAlignmentError(f"duplicate anchor for {shot_id}")
        seen.add(shot_id)
        anchors.append(PlayAnchor(shot_id, int(raw["source_frame"]), str(raw.get("event", "snap"))))
    return PlayAlignment(play_id, tuple(sorted(anchors, key=lambda anchor: anchor.shot_id)))
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_replay.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/football_tracking/replay.py tests/test_replay.py
git commit -m "feat: add reviewed play-time alignment"
```

---

### Task 4: Cross-shot candidate scoring

**Files:**
- Modify: `src/football_tracking/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `PlayAlignment` (Task 3); `TeamEvidence` from `identity.py`.
- Produces:
  - `class FieldTrack` with fields `tracklet_id: str`, `shot_id: str`, `samples: tuple[tuple[float, float, float], ...]` where each sample is `(play_time_s, field_x_yards, field_y_yards)`
  - `cross_shot_candidate_scores(left: Sequence[FieldTrack], right: Sequence[FieldTrack], teams: Mapping[str, TeamEvidence], *, max_field_distance_yards: float = 6.0, min_overlap_samples: int = 5, sample_tolerance_s: float = 0.05) -> dict[tuple[str, str], float]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_replay.py`:

```python
from football_tracking.identity import TeamEvidence
from football_tracking.replay import FieldTrack, cross_shot_candidate_scores


def line(tracklet_id, shot_id, x0, y0, *, dx=1.0, count=10, start=0.0, step=0.1):
    return FieldTrack(
        tracklet_id, shot_id,
        tuple((start + index * step, x0 + index * dx, y0) for index in range(count)),
    )


def teams(**assignment):
    return {key: TeamEvidence(value, 0.95, "rgb_prototype", 20) for key, value in assignment.items()}


def test_matching_field_motion_scores_high_and_distant_motion_is_dropped() -> None:
    left = [line("shot-0:t1", "shot-0", 10.0, 20.0), line("shot-0:t2", "shot-0", 60.0, 40.0)]
    right = [line("shot-1:t1", "shot-1", 10.0, 20.0), line("shot-1:t2", "shot-1", 60.0, 40.0)]
    evidence = teams(**{
        "shot-0:t1": "DET", "shot-0:t2": "LAR", "shot-1:t1": "DET", "shot-1:t2": "LAR",
    })

    scores = cross_shot_candidate_scores(left, right, evidence)

    assert scores[("shot-0:t1", "shot-1:t1")] > 0.9
    assert scores[("shot-0:t2", "shot-1:t2")] > 0.9
    # 50 yards apart, well beyond max_field_distance_yards: no candidate at all.
    assert ("shot-0:t1", "shot-1:t2") not in scores


def test_team_conflict_and_thin_overlap_produce_no_candidate() -> None:
    left = [line("shot-0:t1", "shot-0", 10.0, 20.0)]
    right = [line("shot-1:t1", "shot-1", 10.0, 20.0)]

    conflicting = teams(**{"shot-0:t1": "DET", "shot-1:t1": "LAR"})
    assert cross_shot_candidate_scores(left, right, conflicting) == {}

    unknown = teams(**{"shot-0:t1": "unknown", "shot-1:t1": "DET"})
    assert cross_shot_candidate_scores(left, right, unknown) == {}

    thin_left = [line("shot-0:t1", "shot-0", 10.0, 20.0, count=3)]
    thin_right = [line("shot-1:t1", "shot-1", 10.0, 20.0, count=3)]
    agreeing = teams(**{"shot-0:t1": "DET", "shot-1:t1": "DET"})
    assert cross_shot_candidate_scores(thin_left, thin_right, agreeing) == {}


def test_opposing_trajectory_shape_scores_below_matching_shape() -> None:
    left = [line("shot-0:t1", "shot-0", 10.0, 20.0, dx=1.0)]
    right = [
        line("shot-1:same", "shot-1", 10.0, 20.0, dx=1.0),
        line("shot-1:reverse", "shot-1", 10.0, 20.0, dx=-1.0),
    ]
    evidence = teams(**{"shot-0:t1": "DET", "shot-1:same": "DET", "shot-1:reverse": "DET"})

    scores = cross_shot_candidate_scores(left, right, evidence)

    assert scores[("shot-0:t1", "shot-1:same")] > scores[("shot-0:t1", "shot-1:reverse")]


def test_scores_are_bounded_and_deterministic() -> None:
    left = [line("shot-0:t1", "shot-0", 10.0, 20.0)]
    right = [line("shot-1:t1", "shot-1", 10.4, 20.3)]
    evidence = teams(**{"shot-0:t1": "DET", "shot-1:t1": "DET"})

    first = cross_shot_candidate_scores(left, right, evidence)
    second = cross_shot_candidate_scores(left, right, evidence)

    assert first == second
    assert all(0.0 <= value <= 1.0 for value in first.values())
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_replay.py -k "field or team_conflict or trajectory or bounded" -v
```

Expected: FAIL with `ImportError: cannot import name 'FieldTrack'`.

- [ ] **Step 3: Implement the scorer**

Append to `src/football_tracking/replay.py` (and add `from typing import Mapping, Sequence` plus `import numpy as np` and `from .identity import TeamEvidence` to the imports at the top):

```python
@dataclass(frozen=True, slots=True)
class FieldTrack:
    """A tracklet resampled into (play_time_s, field_x_yards, field_y_yards)."""

    tracklet_id: str
    shot_id: str
    samples: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if any(len(sample) != 3 for sample in self.samples):
            raise ValueError("samples must be (play_time_s, field_x_yards, field_y_yards)")


def _paired_samples(
    left: FieldTrack,
    right: FieldTrack,
    tolerance_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair samples that share a play time within tolerance, nearest match wins."""

    if not left.samples or not right.samples:
        return np.empty((0, 2)), np.empty((0, 2))
    right_times = np.asarray([sample[0] for sample in right.samples], dtype=float)
    right_points = np.asarray([(sample[1], sample[2]) for sample in right.samples], dtype=float)
    left_pairs: list[tuple[float, float]] = []
    right_pairs: list[tuple[float, float]] = []
    for time_s, x, y in left.samples:
        offsets = np.abs(right_times - time_s)
        index = int(np.argmin(offsets))
        if offsets[index] <= tolerance_s:
            left_pairs.append((x, y))
            right_pairs.append(tuple(right_points[index]))
    return np.asarray(left_pairs, dtype=float), np.asarray(right_pairs, dtype=float)


def _shape_agreement(left_points: np.ndarray, right_points: np.ndarray) -> float:
    """Cosine agreement of net displacement, rescaled to [0, 1]."""

    if len(left_points) < 2:
        return 0.0
    left_delta = left_points[-1] - left_points[0]
    right_delta = right_points[-1] - right_points[0]
    left_norm = float(np.linalg.norm(left_delta))
    right_norm = float(np.linalg.norm(right_delta))
    if left_norm < 1e-6 or right_norm < 1e-6:
        # Two stationary players agree on shape but carry no directional evidence.
        return 0.5
    cosine = float(np.dot(left_delta, right_delta) / (left_norm * right_norm))
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def cross_shot_candidate_scores(
    left: Sequence[FieldTrack],
    right: Sequence[FieldTrack],
    teams: Mapping[str, TeamEvidence],
    *,
    max_field_distance_yards: float = 6.0,
    min_overlap_samples: int = 5,
    sample_tolerance_s: float = 0.05,
) -> dict[tuple[str, str], float]:
    """Score cross-shot pairs from view-invariant evidence only.

    A pair that fails a hard constraint is omitted entirely rather than scored
    low, so ``match_tracklets`` reports ``insufficient_evidence`` instead of a
    weak ``same``.
    """

    if max_field_distance_yards <= 0 or min_overlap_samples < 2:
        raise ValueError("invalid cross-shot scoring constraints")
    scores: dict[tuple[str, str], float] = {}
    for left_track in sorted(left, key=lambda track: track.tracklet_id):
        left_team = teams.get(left_track.tracklet_id)
        if left_team is None or left_team.team == "unknown":
            continue
        for right_track in sorted(right, key=lambda track: track.tracklet_id):
            right_team = teams.get(right_track.tracklet_id)
            if right_team is None or right_team.team == "unknown":
                continue
            if left_team.team != right_team.team:
                continue
            left_points, right_points = _paired_samples(left_track, right_track, sample_tolerance_s)
            if len(left_points) < min_overlap_samples:
                continue
            distances = np.linalg.norm(left_points - right_points, axis=1)
            mean_distance = float(np.mean(distances))
            if mean_distance > max_field_distance_yards:
                continue
            position = 1.0 - (mean_distance / max_field_distance_yards)
            shape = _shape_agreement(left_points, right_points)
            team_confidence = float(left_team.score * right_team.score)
            score = 0.55 * position + 0.30 * shape + 0.15 * team_confidence
            scores[(left_track.tracklet_id, right_track.tracklet_id)] = max(0.0, min(1.0, score))
    return scores
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_replay.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/football_tracking/replay.py tests/test_replay.py
git commit -m "feat: score cross-shot candidates from field position and shape"
```

---

### Task 5: Wire cross-shot resolution into the run pipeline

**Files:**
- Modify: `src/football_tracking/cli.py` — `_add_run_arguments` (line 45-65), the identity block from Task 2, the `write_review_json` call (line 378), and the `tracking-evaluation.json` block (lines 383-394)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_play_alignment`, `PlayAlignment`, `FieldTrack`, `cross_shot_candidate_scores` (Tasks 3-4); `match_tracklets`, `stable_anonymous_ids` (`identity.py`); `evaluate_cross_shot_identity`, `load_cross_shot_identity` (Task 1).
- Produces: `--play-alignment PATH` CLI flag; populated `Observation.play_id`; `identity-links.json` with `status` in `{"not_attempted", "abstained", "resolved"}`; a `cross_shot` section in `tracking-evaluation.json`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_run_abstains_without_calibration_and_records_the_reason(tmp_path) -> None:
    from football_tracking.cli import main

    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({
        "reviewed": True, "play_id": "play-1",
        "anchors": [
            {"shot_id": "shot-0", "source_frame": 120, "event": "snap"},
            {"shot_id": "shot-1", "source_frame": 820, "event": "snap"},
        ],
    }), encoding="utf-8")

    output = tmp_path / "run"
    exit_code = main([
        "run", "--input", "data/all-22-lions-rams-sample.mp4", "--output", str(output),
        "--detector", "synthetic", "--tracker", "iou", "--manual-cut", "712",
        "--play-alignment", str(alignment),
    ])
    assert exit_code == 0

    report = json.loads((output / "identity-links.json").read_text(encoding="utf-8"))
    assert report["status"] == "abstained"
    assert report["reason"] == "no calibrated field positions for the aligned shots"
    assert report["cross_shot_player_ids"] == 0

    review = json.loads((output / "review.json").read_text(encoding="utf-8"))
    assert review["cross_view_identity_resolved"] is False

    observations = (output / "observations.csv").read_text(encoding="utf-8").splitlines()
    header = observations[0].split(",")
    play_id_column = header.index("play_id")
    assert observations[1].split(",")[play_id_column] == "play-1"


def test_run_rejects_unreviewed_play_alignment(tmp_path) -> None:
    from football_tracking.cli import main

    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({"play_id": "play-1", "anchors": []}), encoding="utf-8")

    exit_code = main([
        "run", "--input", "data/all-22-lions-rams-sample.mp4", "--output", str(tmp_path / "run"),
        "--detector", "synthetic", "--tracker", "iou", "--manual-cut", "712",
        "--play-alignment", str(alignment),
    ])
    assert exit_code == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_cli.py -k play_alignment -v
```

Expected: FAIL with `unrecognized arguments: --play-alignment` (SystemExit 2 from argparse, before the assertion).

- [ ] **Step 3: Implement the wiring**

In `src/football_tracking/cli.py`, add the flag at the end of `_add_run_arguments`:

```python
    parser.add_argument("--play-alignment", type=Path, default=None, help="reviewed snap-anchor JSON enabling constrained cross-shot identity joins")
```

Add the imports:

```python
from .identity import IdentityLink, TrackletSummary, match_tracklets, resolve_teams, stable_anonymous_ids, team_feature_from_crop
from .replay import FieldTrack, PlayAlignment, ReplayAlignmentError, cross_shot_candidate_scores, load_play_alignment
```

Add `ReplayAlignmentError` to the caught exceptions in `main`:

```python
    except (FileNotFoundError, MemoryError, RuntimeError, ValueError, KeyError, ReplayAlignmentError) as error:
```

`ReplayAlignmentError` subclasses `ValueError`, so this is documentation of intent rather than new behavior; keep it explicit.

Load the alignment early in `run_pipeline`, immediately after `boundaries` and `ranges` are computed (around line 235):

```python
    alignment: PlayAlignment | None = load_play_alignment(args.play_alignment) if args.play_alignment else None
```

Cross-shot resolution needs calibrated field positions, which only exist after projection. So replace the Task 2 identity block and the observation loop that follows it with this ordering: build observations first with a provisional identity map, then resolve, then rewrite the identity fields. Replace the block from `identity_links: list[IdentityLink] = []` through the end of the observation-building `for row in raw_tracks:` loop with:

```python
    identity_links: list[IdentityLink] = []
    link_report: dict[str, Any] = {"status": "not_attempted", "reason": "--play-alignment was not provided"}
    calibrations = _load_calibrations(args.calibration)
    observations: list[Observation] = []
    for row in raw_tracks:
        evidence = teams.get(row.tracklet_id)
        shot_id = row.tracklet_id.split(":", 1)[0]
        observation = Observation(
            run_id=run_id,
            shot_id=shot_id,
            frame_index=row.frame_index,
            pts=row.pts,
            time_base=info.time_base,
            tracklet_id=row.tracklet_id,
            player_id=None,
            bbox_xyxy_px=row.bbox_xyxy_px,
            detection_score=row.score,
            team=evidence.team if evidence else "unknown",
            team_score=evidence.score if evidence else 0.0,
            jersey_number=None,
            field_xy_yards=None,
            position_source=None,
            calibration_id=None,
            identity_version=1,
            play_id=alignment.play_id if alignment and shot_id in alignment.shots() else None,
        )
        homography = calibrations.get(observation.shot_id) or calibrations.get("*")
        if homography is not None:
            observation = project_observation(observation, homography)
        observations.append(observation)
    if alignment is not None:
        field_tracks: dict[str, list[FieldTrack]] = {}
        samples: dict[str, list[tuple[float, float, float]]] = {}
        for observation in observations:
            if observation.field_xy_yards is None:
                continue
            play_time = alignment.play_time_s(observation.shot_id, observation.frame_index, info.source_fps)
            if play_time is None:
                continue
            samples.setdefault(observation.tracklet_id, []).append((play_time, *observation.field_xy_yards))
        for tracklet_id, rows in samples.items():
            shot_id = tracklet_id.split(":", 1)[0]
            field_tracks.setdefault(shot_id, []).append(FieldTrack(tracklet_id, shot_id, tuple(sorted(rows))))
        aligned_shots = [shot for shot in alignment.shots() if field_tracks.get(shot)]
        if len(aligned_shots) < 2:
            link_report = {"status": "abstained", "reason": "no calibrated field positions for the aligned shots"}
        else:
            left_shot, right_shot = aligned_shots[0], aligned_shots[1]
            candidate_scores = cross_shot_candidate_scores(field_tracks[left_shot], field_tracks[right_shot], teams)
            identity_links = match_tracklets(
                [track.tracklet_id for track in field_tracks[left_shot]],
                [track.tracklet_id for track in field_tracks[right_shot]],
                candidate_scores,
            )
            accepted = [link for link in identity_links if link.decision == "same"]
            link_report = {
                "status": "resolved" if accepted else "abstained",
                "reason": "constrained cross-shot match" if accepted else "no candidate pair cleared the threshold and margin",
                "play_id": alignment.play_id,
                "left_shot": left_shot,
                "right_shot": right_shot,
                "candidate_pairs": len(candidate_scores),
                "accepted_links": len(accepted),
            }
    identity_map = stable_anonymous_ids(tracklet_ids, identity_links)
    observations = [
        replace(observation, player_id=identity_map.get(observation.tracklet_id))
        for observation in observations
    ]
    by_player: dict[str, set[str]] = {}
    for tracklet_id, player_id in identity_map.items():
        by_player.setdefault(player_id, set()).add(tracklet_id.split(":", 1)[0])
    link_report.update({
        "shot_count": len(boundaries),
        "tracklet_count": len(tracklet_ids),
        "player_id_count": len(set(identity_map.values())),
        "cross_shot_player_ids": sum(1 for shots in by_player.values() if len(shots) > 1),
        "links": [
            {"left_key": link.left_key, "right_key": link.right_key, "decision": link.decision, "score": link.score}
            for link in identity_links
        ],
    })
```

`Observation` is a frozen dataclass, so the rebuild uses `dataclasses.replace` rather than
`to_dict()` round-tripping, which would turn the tuple fields into lists and fail
`__post_init__`. Add the import to `cli.py`:

```python
from dataclasses import replace
```

Update the `write_review_json` call so resolution status is derived rather than hardcoded:

```python
        resolved = link_report.get("status") == "resolved"
        write_review_json(destination / "review.json", {
            "status": "resolved_cross_view" if resolved else "unresolved_cross_view",
            "shot_count": len(boundaries),
            "cross_view_identity_resolved": resolved,
            "cross_shot_links": link_report.get("accepted_links", 0),
            "notes": [
                "Replay relationship remains unresolved unless manually aligned and linked.",
                "Generic or proxy detector identities are not roster identities.",
            ],
        })
```

Finally, extend the quality report. Replace the `tracking-evaluation.json` block with:

```python
    if args.reviewed_reference is None:
        quality_report = {"status": "not_evaluated", "reason": "--reviewed-reference was not provided"}
    else:
        try:
            quality_report = evaluate_tracking(raw_tracks, load_reviewed_mot_reference(args.reviewed_reference))
            quality_report["reference_path"] = str(args.reviewed_reference)
            quality_report["reference_sha256"] = sha256_file(args.reviewed_reference)
            quality_report["cross_shot"] = evaluate_cross_shot_identity(
                identity_map,
                raw_tracks,
                load_reviewed_mot_reference(args.reviewed_reference),
                load_cross_shot_identity(args.reviewed_reference),
            )
        except EvaluationError as error:
            quality_report = {"status": "not_evaluated", "reason": f"{type(error).__name__}: {error}"}
```

Add the new names to the evaluation import on line 19:

```python
from .evaluation import EvaluationError, evaluate_cross_shot_identity, evaluate_tracking, load_cross_shot_identity, load_reviewed_mot_reference
```

- [ ] **Step 4: Run the full suite to verify it passes**

```bash
UV_CACHE_DIR=.uv-cache uv run pytest -q
```

Expected: PASS, with `tests/test_cli.py` confirming that a run without `--play-alignment` still yields `status: not_attempted` and a 1:1 identity map.

- [ ] **Step 5: Verify the default path is unchanged**

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run \
  --input data/all-22-lions-rams-sample.mp4 --output artifacts/cross-shot-after \
  --detector synthetic --tracker iou --manual-cut 712
python3 -c "
import json
before = json.load(open('artifacts/cross-shot-baseline/identities.json'))
after = json.load(open('artifacts/cross-shot-after/identities.json'))
print('identical default identity map:', before == after)
"
```

Expected: `identical default identity map: True`.

- [ ] **Step 6: Commit**

```bash
git add src/football_tracking/cli.py tests/test_cli.py
git commit -m "feat: resolve cross-shot identity behind reviewed play alignment"
```

---

### Task 6: Record the measured state and the remaining gates

**Files:**
- Create: `docs/evidence/cross-shot-identity.md`
- Modify: `README.md` (after the "Run locally" replay section), `docs/accuracy-calibration-replay.md:96` (the paragraph asserting the CLI always passes `[]`), `docs/evaluation-plan.md:39` (annotate the cross-view row with its measurement command)

**Interfaces:**
- Consumes: `identity-links.json` and the `cross_shot` block of `tracking-evaluation.json` (Tasks 2, 5).
- Produces: no code interfaces.

- [ ] **Step 1: Regenerate the evidence artifacts**

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run \
  --input data/all-22-lions-rams-sample.mp4 \
  --output artifacts/cross-shot-evidence \
  --detector synthetic --tracker iou --manual-cut 712
```

- [ ] **Step 2: Write the evidence document**

Create `docs/evidence/cross-shot-identity.md` containing, in this order: the commit and command used; the measured `identity-links.json` values (`status`, `tracklet_count`, `player_id_count`, `cross_shot_player_ids`); the confirmation that McByte's `masks_active` was `true` on both shots with `fallback_reason: null`; and a "Blocked gates" table copying the three prerequisites from this plan's milestone section with their exact owners and inputs.

State the conclusion plainly: cross-shot identity was never implemented for any tracker, the McByte adapter is not implicated, and the resolver abstains until reviewed anchors and shot-specific calibration landmarks exist.

- [ ] **Step 3: Correct the stale claim in the replay guide**

In `docs/accuracy-calibration-replay.md`, replace the sentence beginning "The current CLI calls `stable_anonymous_ids(tracklet_ids, [])`" with a description of the implemented behavior: the CLI now passes links produced by `replay.cross_shot_candidate_scores` and `identity.match_tracklets` when `--play-alignment` is supplied, abstains without calibrated field positions, and records every decision in `identity-links.json`.

- [ ] **Step 4: Document the command in the README**

Add a "Cross-shot replay identity" subsection showing:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run \
  --input data/all-22-lions-rams-sample.mp4 \
  --output artifacts/cross-shot \
  --detector rfdetr --detector-checkpoint /path/to/football-rfdetr.pth \
  --tracker botsort --manual-cut 712 \
  --calibration /path/to/shot-landmarks.json \
  --play-alignment /path/to/reviewed-snap-anchors.json \
  --reviewed-reference /path/to/reviewed-mot-reference.json
```

State that without `--calibration` for both shots the resolver abstains by design, and that `tracking-evaluation.json` reports `cross_shot.status: not_evaluated` unless the reviewed reference carries a `cross_shot_identity` map.

- [ ] **Step 5: Verify the documented command shape parses**

```bash
UV_CACHE_DIR=.uv-cache uv run python -m football_tracking run --help | grep -E "play-alignment|calibration|reviewed-reference"
git diff --check
```

Expected: all three flags listed; no whitespace errors.

- [ ] **Step 6: Commit**

```bash
git add docs/evidence/cross-shot-identity.md docs/accuracy-calibration-replay.md docs/evaluation-plan.md README.md
git commit -m "docs: record measured cross-shot identity state and blocked gates"
```

---

## Verification checklist for the whole milestone

- [ ] `UV_CACHE_DIR=.uv-cache uv run pytest -q` passes.
- [ ] A run without `--play-alignment` produces an identity map byte-identical to the pre-change baseline.
- [ ] A run with `--play-alignment` but no `--calibration` reports `status: abstained` and merges nothing.
- [ ] An unreviewed alignment or reference file exits non-zero with a named error.
- [ ] `tracking-evaluation.json` reports `cross_shot.status: not_evaluated` when the reference has no `cross_shot_identity` map.
- [ ] `docs/evidence/cross-shot-identity.md` states measured numbers and blocked gates, and claims no accuracy the artifacts do not support.
