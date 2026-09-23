"""Local CVAT video XML bridge for proposals and reviewed annotations."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Mapping, Sequence


class CvatBridgeError(ValueError):
    """Raised when CVAT data cannot be mapped without losing source meaning."""


_OBJECT_LABELS = {"player", "official", "football"}
_ANNOTATION_LABELS = _OBJECT_LABELS | {"calibration_landmark"}
_TAG_LABELS = {"timing_event", "shot_boundary"}


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or isinstance(value, float) and not value.is_integer():
        raise CvatBridgeError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise CvatBridgeError(f"{label} must be an integer") from error
    return result


def _as_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CvatBridgeError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise CvatBridgeError(f"{label} must be finite")
    return result


def _xml_float(value: float) -> str:
    return str(float(value))


def _add_attribute(parent: ET.Element, name: str, value: Any) -> None:
    if value is None or value == "":
        return
    node = ET.SubElement(parent, "attribute", {"name": name})
    node.text = str(value)


def _attribute_values(parent: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for attribute in parent.findall("attribute"):
        name = str(attribute.get("name", "")).strip()
        if not name:
            raise CvatBridgeError("CVAT attributes need a name")
        result[name] = attribute.text or ""
    return result


def _attribute_definition(
    label: ET.Element,
    name: str,
    input_type: str = "text",
    *,
    default: str = "",
    values: Sequence[str] = (),
    mutable: bool = True,
) -> None:
    attribute = ET.SubElement(label.find("attributes"), "attribute")  # type: ignore[arg-type]
    ET.SubElement(attribute, "name").text = name
    ET.SubElement(attribute, "mutable").text = "True" if mutable else "False"
    ET.SubElement(attribute, "input_type").text = input_type
    ET.SubElement(attribute, "default_value").text = default
    ET.SubElement(attribute, "values").text = "\n".join(values)


def _add_cvat_labels(parent: ET.Element, teams: Sequence[str]) -> None:
    common = (
        ("shot_id", "text", "", ()),
        ("source_run_id", "text", "", ()),
        ("source_tracklet_id", "text", "", ()),
        ("source_parent_tracklet_id", "text", "", ()),
        ("proposal_player_id", "text", "", ()),
        ("anonymous_id", "text", "unknown", ()),
        ("team", "select", "unknown", tuple(teams)),
        ("team_score", "number", "", ()),
        ("detection_score", "number", "", ()),
        ("source_pts", "text", "", ()),
        ("play_id", "text", "", ()),
        ("play_time_s", "number", "", ()),
        ("visibility", "select", "visible", ("visible", "partially_visible", "occluded", "out_of_frame", "unknown")),
        ("jersey_readable", "checkbox", "false", ()),
        ("review_status", "select", "unreviewed", ("unreviewed", "reviewed", "accepted", "rejected")),
    )
    landmark = (
        ("landmark_id", "text", "", ()),
        ("field_x_yards", "number", "", ()),
        ("field_y_yards", "number", "", ()),
        ("role", "select", "fit", ("fit", "withheld")),
        ("source_pts", "text", "", ()),
        ("review_status", "select", "unreviewed", ("unreviewed", "reviewed", "accepted", "rejected")),
    )
    event = (
        ("event", "select", "snap", ("snap", "action_start", "release", "contact", "whistle", "end", "corresponding")),
        ("play_id", "text", "", ()),
        ("correspondence_id", "text", "", ()),
        ("play_time_s", "number", "", ()),
        ("source_pts", "text", "", ()),
    )
    boundary = (
        ("shot_id", "text", "", ()),
        ("boundary", "select", "start", ("start",)),
        ("end_frame_exclusive", "text", "", ()),
        ("play_id", "text", "", ()),
        ("split", "text", "unassigned", ()),
        ("camera_label", "text", "unknown", ()),
        ("source_pts", "text", "", ()),
    )
    definitions = {"player": common, "official": common, "football": common, "calibration_landmark": landmark, "timing_event": event, "shot_boundary": boundary}
    for name, attributes in definitions.items():
        label = ET.SubElement(parent, "label")
        ET.SubElement(label, "name").text = name
        attributes_node = ET.SubElement(label, "attributes")
        for attribute_name, input_type, default, values in attributes:
            _attribute_definition(label, attribute_name, input_type, default=default, values=values)


def _normalized_source(source: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = {
            "sha256": str(source["sha256"]),
            "width": _as_int(source["width"], "source width"),
            "height": _as_int(source["height"], "source height"),
            "frame_count": _as_int(source["frame_count"], "source frame_count"),
            "time_base": [_as_int(value, "source time_base") for value in source["time_base"]],
        }
    except (KeyError, TypeError) as error:
        raise CvatBridgeError("source metadata is incomplete") from error
    if not normalized["sha256"] or normalized["width"] <= 0 or normalized["height"] <= 0 or normalized["frame_count"] <= 0:
        raise CvatBridgeError("source metadata must contain a hash and positive dimensions/frame count")
    if len(normalized["time_base"]) != 2 or min(normalized["time_base"]) <= 0:
        raise CvatBridgeError("source time_base must contain two positive integers")
    return normalized


def _normalized_shots(shots: Mapping[str, Mapping[str, Any]], frame_count: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    ordered: list[tuple[str, int, int]] = []
    for shot_id, raw in sorted(shots.items()):
        start = _as_int(raw.get("start_frame"), f"{shot_id} start_frame")
        end = _as_int(raw.get("end_frame"), f"{shot_id} end_frame")
        if not shot_id.strip() or start < 0 or end <= start or end > frame_count:
            raise CvatBridgeError(f"shot {shot_id!r} has an invalid source-frame interval")
        result[str(shot_id)] = {
            "start_frame": start,
            "end_frame": end,
            "play_id": None if raw.get("play_id") in (None, "") else str(raw["play_id"]),
            "split": str(raw.get("split", "unassigned")),
            "camera_label": str(raw.get("camera_label", "unknown")),
        }
        if raw.get("start_pts") is not None:
            result[str(shot_id)]["start_pts"] = _as_int(raw["start_pts"], f"{shot_id} start_pts")
        ordered.append((str(shot_id), start, end))
    if not result:
        raise CvatBridgeError("at least one shot range is required")
    ordered.sort(key=lambda item: (item[1], item[2], item[0]))
    for previous, current in zip(ordered, ordered[1:]):
        if current[1] < previous[2]:
            raise CvatBridgeError(f"shot ranges overlap: {previous[0]} and {current[0]}")
    return result


def build_cvat_preannotations(
    observations: Sequence[Mapping[str, str]],
    *,
    source: Mapping[str, Any],
    run: Mapping[str, Any],
    shots: Mapping[str, Mapping[str, Any]],
    observations_csv_sha256: str,
    pts_by_frame: Sequence[int] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build CVAT for video 1.1 XML and a lossless inference provenance sidecar."""

    normalized_source = _normalized_source(source)
    normalized_shots = _normalized_shots(shots, normalized_source["frame_count"])
    if str(run.get("input_sha256", "")) != normalized_source["sha256"]:
        raise CvatBridgeError("run input sha256 does not match the source video")
    run_id = str(run.get("run_id", "")).strip()
    if not run_id:
        raise CvatBridgeError("run manifest has no run_id")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    provenance_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    observed_teams = {"DET", "LAR", "unknown"}
    for raw_value in observations:
        raw = {str(key): str(value) for key, value in raw_value.items()}
        try:
            shot_id = raw["shot_id"]
            frame = _as_int(raw["frame_index"], "observation frame_index")
            pts = _as_int(raw["pts"], "observation pts")
            tracklet_id = raw["tracklet_id"].strip()
            bbox = tuple(float(value) for value in json.loads(raw["bbox_xyxy_px"]))
            score = _as_float(raw["detection_score"], "observation detection_score")
            team_score = _as_float(raw.get("team_score", "0"), "observation team_score")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise CvatBridgeError(f"invalid observation row: {error}") from error
        key = (shot_id, frame, tracklet_id)
        shot = normalized_shots.get(shot_id)
        if not tracklet_id or key in seen:
            raise CvatBridgeError("observation has an empty tracklet id or a duplicate source frame")
        seen.add(key)
        if shot is None or not shot["start_frame"] <= frame < shot["end_frame"]:
            raise CvatBridgeError(f"observation {shot_id}:{frame} lies outside its declared shot")
        if not 0 <= frame < normalized_source["frame_count"] or pts < 0:
            raise CvatBridgeError(f"observation {shot_id}:{frame} has an invalid source frame or PTS")
        if pts_by_frame is not None and frame < len(pts_by_frame) and pts != int(pts_by_frame[frame]):
            raise CvatBridgeError(f"observation {shot_id}:{frame} PTS does not match the source video")
        if len(bbox) != 4 or not all(math.isfinite(item) for item in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise CvatBridgeError(f"observation {shot_id}:{frame} has an invalid box")
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > normalized_source["width"] or bbox[3] > normalized_source["height"]:
            raise CvatBridgeError(f"observation {shot_id}:{frame} box lies outside the source image")
        if not 0.0 <= score <= 1.0 or not 0.0 <= team_score <= 1.0:
            raise CvatBridgeError(f"observation {shot_id}:{frame} has an invalid confidence")
        if raw.get("run_id") != run_id:
            raise CvatBridgeError(f"observation {shot_id}:{frame} belongs to a different run")
        try:
            row_time_base = tuple(int(item) for item in json.loads(raw["time_base"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise CvatBridgeError(f"observation {shot_id}:{frame} has an invalid time_base") from error
        if row_time_base != tuple(normalized_source["time_base"]):
            raise CvatBridgeError(f"observation {shot_id}:{frame} time_base does not match the source video")
        team = raw.get("team", "unknown") or "unknown"
        observed_teams.add(team)
        value = {"raw_row": raw, "bbox": bbox, "frame": frame, "pts": pts, "team": team, "team_score": team_score, "score": score}
        grouped[(shot_id, tracklet_id)].append(value)
        provenance_rows.append({"shot_id": shot_id, "source_frame": frame, "tracklet_id": tracklet_id, "raw_row": raw})

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    for name, value in (("id", "0"), ("name", f"football-tracking-{run_id}"), ("size", str(normalized_source["frame_count"])), ("mode", "interpolation"), ("overlap", "0"), ("start_frame", "0"), ("stop_frame", str(normalized_source["frame_count"] - 1)), ("frame_filter", ""), ("subset", "default")):
        ET.SubElement(task, name).text = value
    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    ET.SubElement(segment, "id").text = "0"
    ET.SubElement(segment, "start").text = "0"
    ET.SubElement(segment, "stop").text = str(normalized_source["frame_count"] - 1)
    ET.SubElement(segment, "url").text = ""
    owner = ET.SubElement(task, "owner")
    ET.SubElement(owner, "username").text = ""
    ET.SubElement(owner, "email").text = ""
    assignee = ET.SubElement(task, "assignee")
    ET.SubElement(assignee, "username").text = ""
    labels = ET.SubElement(task, "labels")
    _add_cvat_labels(labels, tuple(sorted(observed_teams)))

    # A shot-boundary tag carries the exclusive end frame without shifting any
    # CVAT frame number away from the source video's zero-based frame index.
    for shot_id, shot in sorted(normalized_shots.items(), key=lambda item: (item[1]["start_frame"], item[0])):
        tag = ET.SubElement(root, "tag", {"frame": str(shot["start_frame"]), "label": "shot_boundary"})
        for name, value in (("shot_id", shot_id), ("boundary", "start"), ("end_frame_exclusive", shot["end_frame"]), ("play_id", shot["play_id"]), ("split", shot["split"]), ("camera_label", shot["camera_label"]), ("source_pts", shot.get("start_pts"))):
            _add_attribute(tag, name, value)

    track_id = 0
    for (shot_id, tracklet_id), rows in sorted(grouped.items()):
        track = ET.SubElement(root, "track", {"id": str(track_id), "label": "player", "source": "auto"})
        track_id += 1
        shot = normalized_shots[shot_id]
        for item in sorted(rows, key=lambda value: value["frame"]):
            raw = item["raw_row"]
            x1, y1, x2, y2 = item["bbox"]
            box = ET.SubElement(track, "box", {
                "frame": str(item["frame"]), "xtl": _xml_float(x1), "ytl": _xml_float(y1),
                "xbr": _xml_float(x2), "ybr": _xml_float(y2), "outside": "0", "occluded": "0", "keyframe": "1", "z_order": "0",
            })
            attributes = {
                "shot_id": shot_id,
                "source_run_id": run_id,
                "source_tracklet_id": tracklet_id,
                "source_parent_tracklet_id": raw.get("source_tracklet_id"),
                "proposal_player_id": raw.get("player_id"),
                "anonymous_id": "unknown",
                "team": item["team"],
                "team_score": raw.get("team_score"),
                "detection_score": raw.get("detection_score"),
                "source_pts": item["pts"],
                "play_id": raw.get("play_id") or shot.get("play_id"),
                "play_time_s": raw.get("play_time_s"),
                "visibility": "visible",
                "jersey_readable": "false",
                "review_status": "unreviewed",
            }
            for name, value in attributes.items():
                _add_attribute(box, name, value)

    xml_text = ET.tostring(root, encoding="unicode", xml_declaration=True)
    provenance = {
        "schema_version": 1,
        "cvat_format": "CVAT for video 1.1",
        "source": normalized_source,
        "run": {key: run.get(key) for key in ("run_id", "detector", "detector_version", "tracker", "tracker_version", "config_hash")},
        "observations_csv_sha256": str(observations_csv_sha256),
        "observation_count": len(provenance_rows),
        "shots": normalized_shots,
        "observation_provenance": sorted(provenance_rows, key=lambda item: (item["source_frame"], item["shot_id"], item["tracklet_id"])),
    }
    return xml_text, provenance


def _shot_for_frame(shots: Mapping[str, Mapping[str, Any]], frame: int) -> str:
    matches = [shot_id for shot_id, shot in shots.items() if int(shot["start_frame"]) <= frame < int(shot["end_frame"])]
    if len(matches) != 1:
        raise CvatBridgeError(f"source frame {frame} belongs to {len(matches)} declared shots")
    return matches[0]


def _shape_pts(frame: int, attributes: Mapping[str, str], pts_by_frame: Sequence[int] | None, frame_count: int) -> int:
    if not 0 <= frame < frame_count:
        raise CvatBridgeError(f"CVAT frame {frame} lies outside the source video")
    declared = attributes.get("source_pts")
    source_value: int | None = None
    if pts_by_frame is not None and frame < len(pts_by_frame):
        source_value = _as_int(pts_by_frame[frame], f"source PTS for frame {frame}")
    if declared not in (None, ""):
        declared_value = _as_int(declared, f"CVAT source_pts for frame {frame}")
        if source_value is not None and declared_value != source_value:
            raise CvatBridgeError(f"CVAT PTS for source frame {frame} does not match the source video")
        source_value = declared_value
    if source_value is None or source_value < 0:
        raise CvatBridgeError(f"CVAT frame {frame} has no exact source PTS")
    return source_value


def _review_fields(reviewer: str, reviewed_at: str) -> dict[str, Any]:
    return {"review_status": "reviewed", "reviewer": reviewer, "revision": 1, "reviewed_at": reviewed_at, "annotation_confidence": 1.0}


def import_cvat_manifest(
    xml_text: str,
    provenance: Mapping[str, Any],
    *,
    source_sha256: str,
    pts_by_frame: Sequence[int] | None,
    reviewer: str,
    reviewed_at: str,
) -> dict[str, Any]:
    """Convert reviewed CVAT tracks/tags into the repository annotation manifest."""

    if not reviewer.strip() or not reviewed_at.strip():
        raise CvatBridgeError("reviewer and reviewed_at are required")
    if int(provenance.get("schema_version", 0)) != 1:
        raise CvatBridgeError("CVAT provenance schema_version must be 1")
    source = _normalized_source(provenance.get("source", {}))
    if source["sha256"] != source_sha256:
        raise CvatBridgeError("CVAT provenance source sha256 does not match the input video")
    raw_shots = provenance.get("shots")
    if not isinstance(raw_shots, Mapping):
        raise CvatBridgeError("CVAT provenance has no shot ranges")
    shots = _normalized_shots(raw_shots, source["frame_count"])

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise CvatBridgeError(f"unable to parse CVAT XML: {error}") from error
    if root.tag != "annotations":
        raise CvatBridgeError("expected a CVAT annotations XML root")

    for tag in root.findall("tag"):
        if str(tag.get("label", "")) != "shot_boundary":
            continue
        attributes = _attribute_values(tag)
        shot_id = str(attributes.get("shot_id", "")).strip()
        if not shot_id:
            raise CvatBridgeError("shot_boundary tags need shot_id")
        start_frame = _as_int(tag.get("frame"), "shot boundary start_frame")
        start_pts = _shape_pts(start_frame, attributes, pts_by_frame, source["frame_count"])
        shots[shot_id] = {
            "start_frame": start_frame,
            "end_frame": _as_int(attributes.get("end_frame_exclusive"), "shot boundary end_frame_exclusive"),
            "play_id": attributes.get("play_id") or None,
            "split": attributes.get("split", "unassigned"),
            "camera_label": attributes.get("camera_label", "unknown"),
            "start_pts": start_pts,
        }
    shots = _normalized_shots(shots, source["frame_count"])

    # CVAT exports track properties on each tracked shape. The source converter
    # also emits them there, so track-level attributes are accepted for hand-built
    # or older XML but shape-level values take precedence.
    raw_observations = provenance.get("observation_provenance", [])
    if not isinstance(raw_observations, list):
        raise CvatBridgeError("observation_provenance must be a list")
    provenance_index: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for item in raw_observations:
        if not isinstance(item, Mapping) or not isinstance(item.get("raw_row"), Mapping):
            raise CvatBridgeError("invalid raw observation provenance entry")
        key = (str(item.get("shot_id", "")), _as_int(item.get("source_frame"), "provenance source_frame"), str(item.get("tracklet_id", "")))
        if key in provenance_index:
            raise CvatBridgeError("duplicate raw observation provenance entry")
        provenance_index[key] = item["raw_row"]

    review = _review_fields(reviewer, reviewed_at)
    annotations: list[dict[str, Any]] = []
    landmarks: list[dict[str, Any]] = []
    timing_events: list[dict[str, Any]] = []
    for track in root.findall("track"):
        label = str(track.get("label", ""))
        if label not in _ANNOTATION_LABELS:
            raise CvatBridgeError(f"unsupported CVAT track label {label!r}")
        cvat_id = str(track.get("id", "")).strip()
        if not cvat_id:
            raise CvatBridgeError("CVAT tracks need an id")
        track_attributes = _attribute_values(track)
        track_id = f"cvat-{cvat_id}"
        for shape in list(track):
            if shape.tag not in {"box", "points"}:
                continue
            if shape.get("outside", "0") == "1":
                continue
            attributes = {**track_attributes, **_attribute_values(shape)}
            try:
                frame = _as_int(shape.get("frame"), "CVAT source frame")
            except CvatBridgeError:
                raise
            shot_id = str(attributes.get("shot_id") or _shot_for_frame(shots, frame))
            shot = shots.get(shot_id)
            if shot is None or not int(shot["start_frame"]) <= frame < int(shot["end_frame"]):
                raise CvatBridgeError(f"CVAT annotation at source frame {frame} lies outside shot {shot_id!r}")
            if shot_id != _shot_for_frame(shots, frame):
                raise CvatBridgeError(f"CVAT annotation at source frame {frame} has a mismatched shot_id")
            pts = _shape_pts(frame, attributes, pts_by_frame, source["frame_count"])
            record_id = f"{shot_id}:{frame}:{track_id}"
            if label == "calibration_landmark":
                point_text = str(shape.get("points", ""))
                point_pairs = point_text.split(";")
                point_values = point_pairs[0].split(",") if len(point_pairs) == 1 else []
                if len(point_values) != 2:
                    raise CvatBridgeError(f"calibration landmark {record_id} must contain one image point")
                try:
                    field_point = (_as_float(attributes["field_x_yards"], "field_x_yards"), _as_float(attributes["field_y_yards"], "field_y_yards"))
                except KeyError as error:
                    raise CvatBridgeError(f"calibration landmark {record_id} needs field coordinates") from error
                record = {
                    "id": record_id,
                    "shot_id": shot_id,
                    "source_frame": frame,
                    "pts": pts,
                    "image_xy_px": [_as_float(value, "landmark image coordinate") for value in point_values],
                    "field_xy_yards": list(field_point),
                    "role": attributes.get("role", "fit"),
                    "landmark_id": attributes.get("landmark_id") or None,
                    **review,
                }
                landmarks.append(record)
                continue

            if shape.tag != "box":
                raise CvatBridgeError(f"CVAT {label} track {cvat_id} must use box shapes")
            try:
                bbox = [
                    _as_float(shape.get("xtl"), "box xtl"),
                    _as_float(shape.get("ytl"), "box ytl"),
                    _as_float(shape.get("xbr"), "box xbr"),
                    _as_float(shape.get("ybr"), "box ybr"),
                ]
            except CvatBridgeError:
                raise
            source_tracklet_id = attributes.get("source_tracklet_id", "")
            raw_row = provenance_index.get((shot_id, frame, source_tracklet_id)) if source_tracklet_id else None
            annotation: dict[str, Any] = {
                "id": record_id,
                "shot_id": shot_id,
                "source_frame": frame,
                "pts": pts,
                "bbox_xyxy_px": bbox,
                "track_id": track_id,
                "label": label,
                "team": attributes.get("team", "unknown") or "unknown",
                "visibility": attributes.get("visibility", "occluded" if shape.get("occluded", "0") == "1" else "visible"),
                "occluded": shape.get("occluded", "0") == "1",
                "jersey_readable": attributes.get("jersey_readable", "false").lower() in {"true", "1", "yes"},
                "coordinate_space": "source",
                "proposal_player_id": attributes.get("proposal_player_id") or None,
                "source_tracklet_id": source_tracklet_id or None,
                **review,
            }
            anonymous_id = attributes.get("anonymous_id", attributes.get("global_id", ""))
            if anonymous_id:
                annotation["global_id"] = anonymous_id
            for name in ("team_score", "detection_score", "play_time_s"):
                if attributes.get(name) not in (None, ""):
                    annotation[name] = _as_float(attributes[name], name)
            if raw_row is not None:
                annotation["inference_provenance"] = {
                    "run_id": str(raw_row.get("run_id", "")),
                    "raw_row": dict(raw_row),
                }
            annotations.append(annotation)

    for tag_index, tag in enumerate(root.findall("tag")):
        label = str(tag.get("label", ""))
        attributes = _attribute_values(tag)
        frame = _as_int(tag.get("frame"), f"CVAT tag {label} frame")
        if label == "shot_boundary":
            shot_id = str(attributes.get("shot_id", "")).strip()
            if not shot_id:
                raise CvatBridgeError("shot_boundary tags need shot_id")
            start = frame
            end = _as_int(attributes.get("end_frame_exclusive"), "shot boundary end_frame_exclusive")
            shots[shot_id] = {
                "start_frame": start,
                "end_frame": end,
                "play_id": attributes.get("play_id") or None,
                "split": attributes.get("split", "unassigned"),
                "camera_label": attributes.get("camera_label", "unknown"),
            }
            continue
        if label != "timing_event":
            raise CvatBridgeError(f"unsupported CVAT tag label {label!r}")
        shot_id = str(attributes.get("shot_id") or _shot_for_frame(shots, frame))
        if shot_id not in shots or not int(shots[shot_id]["start_frame"]) <= frame < int(shots[shot_id]["end_frame"]):
            raise CvatBridgeError(f"timing event at source frame {frame} lies outside shot {shot_id!r}")
        event = str(attributes.get("event", "")).strip()
        if not event:
            raise CvatBridgeError("timing_event tags need event")
        timing_event: dict[str, Any] = {
            "id": f"{shot_id}:{frame}:event-{tag_index}",
            "shot_id": shot_id,
            "source_frame": frame,
            "pts": _shape_pts(frame, attributes, pts_by_frame, source["frame_count"]),
            "event": event,
            "play_id": attributes.get("play_id") or None,
            "correspondence_id": attributes.get("correspondence_id") or None,
            **review,
        }
        if attributes.get("play_time_s") not in (None, ""):
            timing_event["play_time_s"] = _as_float(attributes["play_time_s"], "play_time_s")
        timing_events.append(timing_event)

    shots = _normalized_shots(shots, source["frame_count"])
    for collection in (annotations, landmarks, timing_events):
        collection.sort(key=lambda item: (item["source_frame"], item["shot_id"], item["id"]))
    run_info = provenance.get("run")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "reviewed": True,
        "source": source,
        "shots": shots,
        "annotations": annotations,
        "landmarks": landmarks,
        "timing_events": timing_events,
        "frame_labels": [],
        "inference_provenance": {
            "run": dict(run_info) if isinstance(run_info, Mapping) else {},
            "observations_csv_sha256": provenance.get("observations_csv_sha256"),
        },
    }
    return manifest
