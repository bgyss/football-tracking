"""Lossless-enough CVAT video XML interchange with explicit source-frame mapping.

CVAT's video frame numbers are task-local. This module keeps that namespace
separate from original source frames and maps every imported/exported shape
through a source-hashed frame map carrying integer PTS values.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from .metrics import sha256_file
from .field import field_landmark
from .video import VideoInfo, frame_pts


class CvatError(ValueError):
    """Raised when CVAT data cannot be mapped safely to the source video."""


def _positive_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise CvatError(f"{label} must be an integer") from error
    if isinstance(value, bool) or result <= 0:
        raise CvatError(f"{label} must be positive")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise CvatError(f"{label} must be an integer") from error
    if isinstance(value, bool) or result < 0:
        raise CvatError(f"{label} must be non-negative")
    return result


def _bbox(value: Sequence[Any], *, allow_zero: bool = False) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise CvatError("box must contain four numeric coordinates") from error
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise CvatError("box must contain four finite coordinates")
    if allow_zero:
        if values[2] < values[0] or values[3] < values[1]:
            raise CvatError("box coordinates are inverted")
    elif values[2] <= values[0] or values[3] <= values[1]:
        raise CvatError("box must have positive dimensions")
    return values  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class TaskFrame:
    task_frame: int
    source_frame: int
    source_pts: int
    task_width: int
    task_height: int
    crop_xyxy_px: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for value, label in ((self.task_frame, "task_frame"), (self.source_frame, "source_frame"), (self.source_pts, "source_pts")):
            _nonnegative_int(value, label)
        _positive_int(self.task_width, "task_width")
        _positive_int(self.task_height, "task_height")
        crop = _bbox(self.crop_xyxy_px)
        if crop[0] < 0 or crop[1] < 0:
            raise CvatError("crop origin must be non-negative")


@dataclass(frozen=True, slots=True)
class TaskFrameMap:
    source_sha256: str
    source_width: int
    source_height: int
    source_frame_count: int
    time_base: tuple[int, int]
    frames: tuple[TaskFrame, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise CvatError("task frame map schema_version must be 1")
        if len(self.source_sha256) != 64 or any(character not in "0123456789abcdef" for character in self.source_sha256.lower()):
            raise CvatError("source_sha256 must be a 64-character hexadecimal digest")
        _positive_int(self.source_width, "source_width")
        _positive_int(self.source_height, "source_height")
        _positive_int(self.source_frame_count, "source_frame_count")
        if len(self.time_base) != 2 or any(_positive_int(item, "time_base value") <= 0 for item in self.time_base):
            raise CvatError("time_base must contain two positive integers")
        if not self.frames:
            raise CvatError("task frame map must contain at least one frame")
        task_frames = [record.task_frame for record in self.frames]
        source_frames = [record.source_frame for record in self.frames]
        pts_values = [record.source_pts for record in self.frames]
        if task_frames != list(range(len(self.frames))):
            raise CvatError("task_frame values must be dense and start at zero")
        if any(frame >= self.source_frame_count for frame in source_frames):
            raise CvatError("source frame lies outside source video")
        if any(right <= left for left, right in zip(source_frames, source_frames[1:])):
            raise CvatError("source_frame mapping must be strictly increasing and unique")
        if any(right <= left for left, right in zip(pts_values, pts_values[1:])):
            raise CvatError("source_pts mapping must be strictly increasing")
        task_sizes = {(record.task_width, record.task_height) for record in self.frames}
        if len(task_sizes) != 1:
            raise CvatError("all CVAT task frames must use the same dimensions")
        for record in self.frames:
            crop = record.crop_xyxy_px
            if crop[2] > self.source_width or crop[3] > self.source_height:
                raise CvatError("crop lies outside source image")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": {
                "sha256": self.source_sha256,
                "width": self.source_width,
                "height": self.source_height,
                "frame_count": self.source_frame_count,
                "time_base": list(self.time_base),
            },
            "frames": [
                {
                    "task_frame": item.task_frame,
                    "source_frame": item.source_frame,
                    "source_pts": item.source_pts,
                    "task_width": item.task_width,
                    "task_height": item.task_height,
                    "crop_xyxy_px": list(item.crop_xyxy_px),
                }
                for item in self.frames
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, source_sha256: str | None = None) -> "TaskFrameMap":
        if not isinstance(value, Mapping) or not isinstance(value.get("source"), Mapping):
            raise CvatError("task frame map must include source metadata")
        source = value["source"]
        declared_hash = str(source.get("sha256", ""))
        if source_sha256 is not None and declared_hash != source_sha256:
            raise CvatError("task frame map source sha256 does not match the input video")
        raw_frames = value.get("frames")
        if not isinstance(raw_frames, list):
            raise CvatError("task frame map frames must be a list")
        try:
            frames = tuple(
                TaskFrame(
                    task_frame=int(raw["task_frame"]),
                    source_frame=int(raw["source_frame"]),
                    source_pts=int(raw["source_pts"]),
                    task_width=int(raw["task_width"]),
                    task_height=int(raw["task_height"]),
                    crop_xyxy_px=tuple(float(item) for item in raw["crop_xyxy_px"]),  # type: ignore[arg-type]
                )
                for raw in raw_frames
            )
            time_base = tuple(int(item) for item in source["time_base"])
            return cls(
                source_sha256=declared_hash,
                source_width=int(source["width"]),
                source_height=int(source["height"]),
                source_frame_count=int(source["frame_count"]),
                time_base=time_base,  # type: ignore[arg-type]
                frames=frames,
                schema_version=int(value.get("schema_version", 0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, CvatError):
                raise
            raise CvatError(f"invalid task frame map: {error}") from error

    def by_task_frame(self) -> dict[int, TaskFrame]:
        return {item.task_frame: item for item in self.frames}

    def by_source_frame(self) -> dict[int, TaskFrame]:
        return {item.source_frame: item for item in self.frames}


def build_task_frame_map(
    source_path: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    crop_xyxy_px: Sequence[float] | None = None,
    task_size: tuple[int, int] | None = None,
) -> TaskFrameMap:
    """Build an exact task-to-source map using ffprobe's source PTS sequence."""

    source = Path(source_path)
    info = VideoInfo.from_path(source)
    start = _nonnegative_int(start_frame, "start_frame")
    end = info.frame_count if end_frame is None else _positive_int(end_frame, "end_frame")
    if not 0 <= start < end <= info.frame_count:
        raise CvatError("task source interval must satisfy 0 <= start < end <= frame_count")
    pts_values = frame_pts(source)
    if len(pts_values) != info.frame_count:
        raise CvatError(f"exact source PTS coverage is incomplete: {len(pts_values)} PTS values for {info.frame_count} frames")
    crop = (0.0, 0.0, float(info.width), float(info.height)) if crop_xyxy_px is None else _bbox(crop_xyxy_px)
    if crop[0] < 0 or crop[1] < 0 or crop[2] > info.width or crop[3] > info.height:
        raise CvatError("crop rectangle lies outside source image")
    if task_size is None:
        task_width = max(1, int(round(crop[2] - crop[0])))
        task_height = max(1, int(round(crop[3] - crop[1])))
    else:
        if len(task_size) != 2:
            raise CvatError("task_size must contain width and height")
        task_width = _positive_int(task_size[0], "task width")
        task_height = _positive_int(task_size[1], "task height")
    frame_map = tuple(
        TaskFrame(index, source_frame, pts_values[source_frame], task_width, task_height, crop)
        for index, source_frame in enumerate(range(start, end))
    )
    return TaskFrameMap(sha256_file(source), info.width, info.height, info.frame_count, info.time_base, frame_map)


def source_bbox_to_task(bbox_xyxy_px: Sequence[float], frame: TaskFrame) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _bbox(bbox_xyxy_px)
    cx1, cy1, cx2, cy2 = frame.crop_xyxy_px
    crop_width, crop_height = cx2 - cx1, cy2 - cy1
    if crop_width <= 0 or crop_height <= 0:
        raise CvatError("crop must have positive dimensions")
    sx, sy = frame.task_width / crop_width, frame.task_height / crop_height
    result = ((x1 - cx1) * sx, (y1 - cy1) * sy, (x2 - cx1) * sx, (y2 - cy1) * sy)
    if result[0] < -1e-6 or result[1] < -1e-6 or result[2] > frame.task_width + 1e-6 or result[3] > frame.task_height + 1e-6:
        raise CvatError("source box lies outside the task crop")
    return result


def task_bbox_to_source(bbox_xyxy_px: Sequence[float], frame: TaskFrame) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _bbox(bbox_xyxy_px)
    cx1, cy1, cx2, cy2 = frame.crop_xyxy_px
    sx, sy = (cx2 - cx1) / frame.task_width, (cy2 - cy1) / frame.task_height
    result = (cx1 + x1 * sx, cy1 + y1 * sy, cx1 + x2 * sx, cy1 + y2 * sy)
    return tuple(round(item, 6) for item in result)  # type: ignore[return-value]


def source_point_to_task(point_xy_px: Sequence[float], frame: TaskFrame) -> tuple[float, float]:
    if len(point_xy_px) != 2:
        raise CvatError("point must contain x and y coordinates")
    x, y = (float(value) for value in point_xy_px)
    if not math.isfinite(x) or not math.isfinite(y):
        raise CvatError("point coordinates must be finite")
    cx1, cy1, cx2, cy2 = frame.crop_xyxy_px
    if not cx1 <= x <= cx2 or not cy1 <= y <= cy2:
        raise CvatError("source point lies outside the task crop")
    return ((x - cx1) * frame.task_width / (cx2 - cx1), (y - cy1) * frame.task_height / (cy2 - cy1))


def task_point_to_source(point_xy_px: Sequence[float], frame: TaskFrame) -> tuple[float, float]:
    if len(point_xy_px) != 2:
        raise CvatError("point must contain x and y coordinates")
    x, y = (float(value) for value in point_xy_px)
    if not math.isfinite(x) or not math.isfinite(y) or not 0.0 <= x <= frame.task_width or not 0.0 <= y <= frame.task_height:
        raise CvatError("task point lies outside task image")
    cx1, cy1, cx2, cy2 = frame.crop_xyxy_px
    return (round(cx1 + x * (cx2 - cx1) / frame.task_width, 6), round(cy1 + y * (cy2 - cy1) / frame.task_height, 6))


@dataclass(frozen=True, slots=True)
class CvatBox:
    task_frame: int
    bbox_xyxy_px: tuple[float, float, float, float] | None
    outside: bool = False
    occluded: bool = False
    keyframe: bool = True
    attributes: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        _nonnegative_int(self.task_frame, "box task_frame")
        if not self.outside and self.bbox_xyxy_px is None:
            raise CvatError("visible CVAT box needs coordinates")
        if self.outside and self.bbox_xyxy_px is not None:
            _bbox(self.bbox_xyxy_px, allow_zero=True)


@dataclass(frozen=True, slots=True)
class CvatPoint:
    task_frame: int
    points_xy_px: tuple[tuple[float, float], ...] = ()
    outside: bool = False
    occluded: bool = False
    keyframe: bool = True
    attributes: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        _nonnegative_int(self.task_frame, "point task_frame")
        if not self.outside and not self.points_xy_px:
            raise CvatError("visible CVAT point shape needs at least one point")
        if any(len(point) != 2 or not all(math.isfinite(float(value)) for value in point) for point in self.points_xy_px):
            raise CvatError("CVAT point shape contains invalid coordinates")


@dataclass(frozen=True, slots=True)
class CvatTrack:
    track_id: int
    label: str
    source: str
    attributes: Mapping[str, str]
    boxes: tuple[CvatBox, ...]
    points: tuple[CvatPoint, ...] = ()

    def __post_init__(self) -> None:
        _nonnegative_int(self.track_id, "track_id")
        if not self.label:
            raise CvatError("track label must be non-empty")
        if self.source not in {"manual", "auto"}:
            raise CvatError("track source must be manual or auto")
        frames = [box.task_frame for box in self.boxes]
        if frames != sorted(set(frames)):
            raise CvatError("track boxes must have unique ascending task_frame values")
        point_frames = [point.task_frame for point in self.points]
        if point_frames != sorted(set(point_frames)):
            raise CvatError("track point shapes must have unique ascending task_frame values")
        if self.boxes and self.points:
            raise CvatError("CVAT track cannot mix box and point shapes")


@dataclass(frozen=True, slots=True)
class CvatTag:
    task_frame: int
    label: str
    attributes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CvatAnnotations:
    tracks: tuple[CvatTrack, ...]
    tags: tuple[CvatTag, ...] = ()
    task_size: int | None = None
    task_width: int | None = None
    task_height: int | None = None


def _read_attributes(parent: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for child in parent.findall("attribute"):
        name = child.attrib.get("name")
        if name:
            if name in result:
                raise CvatError(f"duplicate CVAT attribute {name!r}")
            result[name] = child.text or ""
    return result


def _import_review_status(attributes: Mapping[str, str], *, reviewed: bool, label: str, frame: int) -> str | None:
    status = str(attributes.get("review_status", "")).strip()
    if status == "rejected":
        return None
    if not reviewed:
        return "unreviewed"
    if status not in {"reviewed", "accepted"}:
        raise CvatError(f"CVAT {label} at source frame {frame} must be reviewed or accepted before promotion")
    return status


def parse_cvat_video_xml(xml_bytes: bytes | str) -> CvatAnnotations:
    """Parse CVAT video track XML 1.1 into deterministic typed records."""

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as error:
        raise CvatError(f"invalid CVAT XML: {error}") from error
    if root.tag != "annotations":
        raise CvatError("CVAT XML root must be <annotations>")
    version = (root.findtext("version") or "").strip()
    if version not in {"1.1", "1.2"}:
        raise CvatError(f"unsupported CVAT video XML version {version!r}")
    tracks: list[CvatTrack] = []
    track_ids: set[int] = set()
    for raw_track in root.findall("track"):
        try:
            track_id = _nonnegative_int(raw_track.attrib["id"], "track id")
        except KeyError as error:
            raise CvatError("CVAT track has no id") from error
        if track_id in track_ids:
            raise CvatError(f"duplicate CVAT track id {track_id}")
        track_ids.add(track_id)
        label = str(raw_track.attrib.get("label", ""))
        source = str(raw_track.attrib.get("source", "manual"))
        track_attributes = _read_attributes(raw_track)
        boxes: list[CvatBox] = []
        seen_frames: set[int] = set()
        for raw_box in raw_track.findall("box"):
            try:
                frame_index = _nonnegative_int(raw_box.attrib["frame"], "box frame")
                outside = raw_box.attrib.get("outside", "0") == "1"
                occluded = raw_box.attrib.get("occluded", "0") == "1"
                keyframe = raw_box.attrib.get("keyframe", "1") == "1"
                coordinates = tuple(float(raw_box.attrib[key]) for key in ("xtl", "ytl", "xbr", "ybr"))
            except (KeyError, TypeError, ValueError) as error:
                raise CvatError(f"invalid box in CVAT track {track_id}") from error
            if frame_index in seen_frames:
                raise CvatError(f"duplicate CVAT box at track/frame {track_id}:{frame_index}")
            seen_frames.add(frame_index)
            if outside:
                box = None
            else:
                box = _bbox(coordinates)
            boxes.append(CvatBox(frame_index, box, outside, occluded, keyframe, _read_attributes(raw_box)))
        points: list[CvatPoint] = []
        for raw_points in raw_track.findall("points"):
            try:
                frame_index = _nonnegative_int(raw_points.attrib["frame"], "points frame")
                outside = raw_points.attrib.get("outside", "0") == "1"
                occluded = raw_points.attrib.get("occluded", "0") == "1"
                keyframe = raw_points.attrib.get("keyframe", "1") == "1"
                raw_pairs = [entry for entry in raw_points.attrib.get("points", "").split(";") if entry]
                coordinates = tuple(tuple(float(value) for value in pair.split(",", 1)) for pair in raw_pairs)
            except (KeyError, TypeError, ValueError) as error:
                raise CvatError(f"invalid points shape in CVAT track {track_id}") from error
            if not outside and (not coordinates or any(len(point) != 2 for point in coordinates)):
                raise CvatError(f"CVAT points shape in track {track_id} needs x,y coordinates")
            points.append(CvatPoint(frame_index, coordinates, outside, occluded, keyframe, _read_attributes(raw_points)))
        tracks.append(CvatTrack(track_id, label, source, track_attributes, tuple(sorted(boxes, key=lambda item: item.task_frame)), tuple(sorted(points, key=lambda item: item.task_frame))))
    tags: list[CvatTag] = []
    for raw_tag in root.findall("tag"):
        try:
            frame_index = _nonnegative_int(raw_tag.attrib["frame"], "tag frame")
        except (KeyError, TypeError) as error:
            raise CvatError("CVAT tag has no valid frame") from error
        tags.append(CvatTag(frame_index, str(raw_tag.attrib.get("label", "")), _read_attributes(raw_tag)))
    task_meta = root.find("./meta/task")
    task_size = None
    task_width = None
    task_height = None
    if task_meta is not None:
        text_value = task_meta.findtext("size")
        if text_value not in (None, ""):
            task_size = _positive_int(text_value, "CVAT task size")
        original_size = task_meta.find("original_size")
        if original_size is not None:
            try:
                task_width = _positive_int(original_size.findtext("width"), "CVAT task width")
                task_height = _positive_int(original_size.findtext("height"), "CVAT task height")
            except CvatError:
                raise
    return CvatAnnotations(tuple(sorted(tracks, key=lambda item: item.track_id)), tuple(sorted(tags, key=lambda item: (item.task_frame, item.label))), task_size, task_width, task_height)


def _append_attributes(parent: ET.Element, values: Mapping[str, Any]) -> None:
    for key, value in sorted(values.items()):
        item = ET.SubElement(parent, "attribute", {"name": str(key)})
        item.text = str(value)


def write_cvat_video_xml(
    tracks: Iterable[CvatTrack],
    frame_map: TaskFrameMap,
    *,
    task_name: str = "football-tracking-proposals",
    labels: Sequence[str] = ("player", "official", "football", "field_landmark", "ground_contact", "timing_event"),
    label_attributes: Mapping[str, Sequence[str]] | None = None,
    point_labels: Sequence[str] = ("field_landmark", "ground_contact", "timing_event"),
    tags: Iterable[CvatTag] = (),
) -> bytes:
    """Serialize CVAT video XML 1.1 and reject shapes outside the frame map."""

    selected_tracks = tuple(sorted(tracks, key=lambda item: item.track_id))
    if len({item.track_id for item in selected_tracks}) != len(selected_tracks):
        raise CvatError("CVAT track ids must be unique")
    frame_records_by_task = frame_map.by_task_frame()
    task_frame_ids = set(frame_records_by_task)
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    for key, value in (("id", 0), ("name", task_name), ("size", len(frame_map.frames)), ("mode", "interpolation"), ("overlap", 0), ("bugtracker", ""), ("created", ""), ("updated", ""), ("subset", "default"), ("start_frame", 0), ("stop_frame", len(frame_map.frames) - 1), ("frame_filter", "")):
        ET.SubElement(task, key).text = str(value)
    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    for key, value in (("id", 0), ("start", 0), ("stop", len(frame_map.frames) - 1), ("url", "")):
        ET.SubElement(segment, key).text = str(value)
    owner = ET.SubElement(task, "owner")
    ET.SubElement(owner, "username").text = ""
    ET.SubElement(owner, "email").text = ""
    original_size = ET.SubElement(task, "original_size")
    ET.SubElement(original_size, "width").text = str(frame_map.frames[0].task_width)
    ET.SubElement(original_size, "height").text = str(frame_map.frames[0].task_height)
    label_root = ET.SubElement(task, "labels")
    for label_name in sorted(set(labels)):
        label = ET.SubElement(label_root, "label")
        ET.SubElement(label, "name").text = label_name
        ET.SubElement(label, "color").text = "#ffffff"
        ET.SubElement(label, "type").text = "points" if label_name in point_labels else "any"
        attribute_names = sorted({
            key
            for track_data in selected_tracks
            if track_data.label == label_name
            for key in (
                set(track_data.attributes)
                | {attribute for box in track_data.boxes for attribute in (box.attributes or {})}
            )
        } | {str(name) for name in (label_attributes or {}).get(label_name, ())})
        label_attributes_node = ET.SubElement(label, "attributes")
        for attribute_name in attribute_names:
            attribute = ET.SubElement(label_attributes_node, "attribute")
            ET.SubElement(attribute, "name").text = attribute_name
            ET.SubElement(attribute, "mutable").text = "True"
            ET.SubElement(attribute, "input_type").text = "text"
            ET.SubElement(attribute, "default_value").text = ""
            ET.SubElement(attribute, "values").text = ""
    for track_data in selected_tracks:
        if track_data.label not in labels:
            raise CvatError(f"track {track_data.track_id} uses undeclared label {track_data.label!r}")
        track = ET.SubElement(root, "track", {"id": str(track_data.track_id), "label": track_data.label, "source": track_data.source})
        _append_attributes(track, track_data.attributes)
        for box_data in track_data.boxes:
            if box_data.task_frame not in task_frame_ids:
                raise CvatError(f"CVAT track {track_data.track_id} references unmapped task frame {box_data.task_frame}")
            values: dict[str, str] = {"frame": str(box_data.task_frame), "outside": "1" if box_data.outside else "0", "occluded": "1" if box_data.occluded else "0", "keyframe": "1" if box_data.keyframe else "0", "z_order": "0"}
            if box_data.bbox_xyxy_px is not None:
                coordinates = _bbox(box_data.bbox_xyxy_px, allow_zero=box_data.outside)
                values.update({key: f"{value:.6f}" for key, value in zip(("xtl", "ytl", "xbr", "ybr"), coordinates)})
            elif box_data.outside:
                values.update({"xtl": "0", "ytl": "0", "xbr": "0", "ybr": "0"})
            shape = ET.SubElement(track, "box", values)
            _append_attributes(shape, box_data.attributes or {})
        for point_data in track_data.points:
            if point_data.task_frame not in task_frame_ids:
                raise CvatError(f"CVAT track {track_data.track_id} references unmapped task frame {point_data.task_frame}")
            values = {"frame": str(point_data.task_frame), "outside": "1" if point_data.outside else "0", "occluded": "1" if point_data.occluded else "0", "keyframe": "1" if point_data.keyframe else "0", "z_order": "0"}
            if point_data.points_xy_px:
                task_frame = frame_records_by_task[point_data.task_frame]
                if any(not 0.0 <= x <= task_frame.task_width or not 0.0 <= y <= task_frame.task_height for x, y in point_data.points_xy_px):
                    raise CvatError(f"CVAT track {track_data.track_id} has a point outside task frame {point_data.task_frame}")
                values["points"] = ";".join(f"{x:.6f},{y:.6f}" for x, y in point_data.points_xy_px)
            elif point_data.outside:
                values["points"] = "0,0"
            shape = ET.SubElement(track, "points", values)
            _append_attributes(shape, point_data.attributes or {})
    for tag_data in sorted(tags, key=lambda item: (item.task_frame, item.label)):
        if tag_data.task_frame not in task_frame_ids:
            raise CvatError(f"CVAT tag references unmapped task frame {tag_data.task_frame}")
        tag = ET.SubElement(root, "tag", {"label": tag_data.label, "frame": str(tag_data.task_frame)})
        _append_attributes(tag, tag_data.attributes)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def read_cvat_bundle(path: str | Path, frame_map_path: str | Path | None = None, *, source_sha256: str | None = None) -> tuple[bytes, TaskFrameMap]:
    """Read CVAT XML and its frame map from an archive or separate files."""

    source = Path(path)
    if zipfile.is_zipfile(source):
        try:
            with zipfile.ZipFile(source) as archive:
                xml_names = [name for name in archive.namelist() if Path(name).name == "annotations.xml"]
                if len(xml_names) != 1:
                    raise CvatError("CVAT archive must contain exactly one annotations.xml")
                xml_bytes = archive.read(xml_names[0])
                if frame_map_path is None:
                    map_names = [name for name in archive.namelist() if Path(name).name == "task-frame-map.json"]
                    if len(map_names) != 1:
                        raise CvatError("CVAT archive must contain exactly one task-frame-map.json")
                    map_value = json.loads(archive.read(map_names[0]))
                else:
                    map_value = json.loads(Path(frame_map_path).read_text(encoding="utf-8"))
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as error:
            raise CvatError(f"unable to read CVAT bundle: {error}") from error
    else:
        if frame_map_path is None:
            raise CvatError("a separate task frame map is required when input is XML")
        try:
            xml_bytes = source.read_bytes()
            map_value = json.loads(Path(frame_map_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CvatError(f"unable to read CVAT XML or frame map: {error}") from error
    return xml_bytes, TaskFrameMap.from_dict(map_value, source_sha256=source_sha256)


def write_cvat_bundle(path: str | Path, xml_bytes: bytes, frame_map: TaskFrameMap) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("annotations.xml", xml_bytes)
        archive.writestr("task-frame-map.json", json.dumps(frame_map.to_dict(), indent=2, sort_keys=True) + "\n")


def manifest_annotations_from_cvat(
    annotations: CvatAnnotations,
    frame_map: TaskFrameMap,
    *,
    shot_id: str,
    shot_range: tuple[int, int],
    reviewed: bool = False,
    reviewer: str | None = None,
    revision: int | None = None,
    reviewed_at: str | None = None,
    annotation_confidence: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert CVAT player tracks and frame tags into manifest records."""

    if not shot_id:
        raise CvatError("shot_id must be non-empty")
    start_frame, end_frame = shot_range
    if start_frame < 0 or end_frame <= start_frame:
        raise CvatError("shot range must be a non-empty source-frame interval")
    if reviewed:
        if not reviewer or not reviewed_at or revision is None or annotation_confidence is None:
            raise CvatError("reviewed import requires reviewer, revision, reviewed_at, and annotation_confidence")
        if not math.isfinite(float(annotation_confidence)) or not 0.0 <= float(annotation_confidence) <= 1.0:
            raise CvatError("annotation_confidence must be between zero and one")
        _positive_int(revision, "revision")
    source_by_task = frame_map.by_task_frame()
    source_by_frame = frame_map.by_source_frame()
    records: list[dict[str, Any]] = []
    landmarks: list[dict[str, Any]] = []
    frame_labels: list[dict[str, Any]] = []
    point_proposals: list[dict[str, Any]] = []
    timing_events: list[dict[str, Any]] = []
    contact_points: dict[tuple[str, int], tuple[tuple[float, float], float | None, dict[str, str], str]] = {}
    for track in annotations.tracks:
        tracklet_id = str(track.attributes.get("tracklet_id") or track.attributes.get("source_tracklet_id") or f"cvat-track-{track.track_id}")
        if track.points:
            for point_shape in track.points:
                if point_shape.task_frame not in source_by_task:
                    raise CvatError(f"CVAT point references unmapped task frame {point_shape.task_frame}")
                mapped = source_by_task[point_shape.task_frame]
                if not start_frame <= mapped.source_frame < end_frame:
                    raise CvatError(f"CVAT point frame {mapped.source_frame} lies outside declared shot {shot_id}")
                # Only explicit visible points can define timing anchors or
                # calibration landmarks. CVAT may carry stale interpolated
                # attributes on keyframe=0 shapes, so discard these before
                # interpreting per-shape metadata.
                if not point_shape.keyframe or point_shape.outside:
                    continue
                attributes = {**track.attributes, **(point_shape.attributes or {})}
                if attributes.get("source_sha256") not in (None, "", frame_map.source_sha256):
                    raise CvatError(f"CVAT point track {track.track_id} source hash does not match task frame map")
                for key, expected in (("source_frame", mapped.source_frame), ("source_pts", mapped.source_pts)):
                    if attributes.get(key) not in (None, ""):
                        try:
                            actual = int(attributes[key])
                        except (TypeError, ValueError) as error:
                            raise CvatError(f"invalid {key} attribute on point track {track.track_id}") from error
                        if actual != expected:
                            raise CvatError(f"CVAT {key} attribute disagrees with task frame map at task frame {point_shape.task_frame}")
                review_status = _import_review_status(attributes, reviewed=reviewed, label=track.label, frame=mapped.source_frame)
                if review_status is None:
                    continue
                if point_shape.outside or len(point_shape.points_xy_px) != 1:
                    point_proposals.append({"shot_id": shot_id, "source_frame": mapped.source_frame, "pts": mapped.source_pts, "label": track.label, "outside": point_shape.outside, "review_status": "unreviewed", "attributes": dict(sorted(attributes.items()))})
                    continue
                source_point = task_point_to_source(point_shape.points_xy_px[0], mapped)
                if not 0 <= source_point[0] <= frame_map.source_width or not 0 <= source_point[1] <= frame_map.source_height:
                    raise CvatError(f"imported point lies outside source image at frame {mapped.source_frame}")
                if track.label == "timing_event":
                    event = str(attributes.get("event", "")).strip()
                    if not event:
                        point_proposals.append({"shot_id": shot_id, "source_frame": mapped.source_frame, "pts": mapped.source_pts, "label": track.label, "image_xy_px": list(source_point), "review_status": "unreviewed", "attributes": dict(sorted(attributes.items()))})
                        continue
                    timing_event: dict[str, Any] = {
                        "id": f"{shot_id}:{mapped.source_frame}:event-{track.track_id}",
                        "shot_id": shot_id,
                        "source_frame": mapped.source_frame,
                        "pts": mapped.source_pts,
                        "event": event,
                        "play_id": attributes.get("play_id") or None,
                        "correspondence_id": attributes.get("correspondence_id") or None,
                        "review_status": review_status,
                    }
                    if attributes.get("play_time_s") not in (None, ""):
                        try:
                            play_time = float(attributes["play_time_s"])
                        except (TypeError, ValueError) as error:
                            raise CvatError(f"invalid play_time_s on CVAT track {track.track_id}") from error
                        if not math.isfinite(play_time):
                            raise CvatError(f"invalid play_time_s on CVAT track {track.track_id}")
                        timing_event["play_time_s"] = play_time
                    if reviewed:
                        timing_event.update({"reviewer": reviewer, "revision": int(revision), "reviewed_at": reviewed_at, "annotation_confidence": float(annotation_confidence)})
                    timing_events.append(timing_event)
                elif track.label == "ground_contact":
                    declared_tracklet_id = str(attributes.get("tracklet_id") or attributes.get("source_tracklet_id") or "").strip()
                    if not declared_tracklet_id:
                        point_proposals.append({"shot_id": shot_id, "source_frame": mapped.source_frame, "pts": mapped.source_pts, "label": track.label, "image_xy_px": list(source_point), "review_status": "unreviewed", "attributes": dict(sorted(attributes.items()))})
                        continue
                    tracklet_id = declared_tracklet_id
                    confidence_value = attributes.get("ground_contact_confidence")
                    confidence = None
                    if confidence_value not in (None, ""):
                        try:
                            confidence = float(confidence_value)
                        except ValueError as error:
                            raise CvatError(f"invalid ground_contact_confidence on track {track.track_id}") from error
                        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                            raise CvatError("ground_contact_confidence must be between zero and one")
                    contact_key = (tracklet_id, mapped.source_frame)
                    if contact_key in contact_points:
                        raise CvatError(f"duplicate ground-contact point for {tracklet_id}@{mapped.source_frame}")
                    contact_points[contact_key] = (source_point, confidence, dict(sorted(attributes.items())), review_status)
                elif track.label == "field_landmark":
                    landmark_id = str(attributes.get("landmark_id", "")).strip()
                    role = str(attributes.get("role", "")).strip()
                    if not landmark_id or role not in {"fit", "withheld"}:
                        point_proposals.append({"shot_id": shot_id, "source_frame": mapped.source_frame, "pts": mapped.source_pts, "label": track.label, "image_xy_px": list(source_point), "review_status": "unreviewed", "attributes": dict(sorted(attributes.items()))})
                        continue
                    try:
                        field_point = field_landmark(landmark_id)
                    except ValueError as error:
                        raise CvatError(f"invalid semantic landmark_id {landmark_id!r} on CVAT track {track.track_id}") from error
                    if attributes.get("field_x_yards") not in (None, "") or attributes.get("field_y_yards") not in (None, ""):
                        try:
                            declared_field = (float(attributes["field_x_yards"]), float(attributes["field_y_yards"]))
                        except (KeyError, TypeError, ValueError) as error:
                            raise CvatError(f"landmark field coordinate attributes are incomplete on CVAT track {track.track_id}") from error
                        if not all(math.isfinite(value) for value in declared_field) or math.dist(declared_field, field_point) > 0.01:
                            raise CvatError(f"landmark field coordinates disagree with semantic ID {landmark_id}")
                    landmark: dict[str, Any] = {"id": f"{shot_id}:{landmark_id}:{mapped.source_frame}", "shot_id": shot_id, "source_frame": mapped.source_frame, "pts": mapped.source_pts, "image_xy_px": list(source_point), "field_xy_yards": list(field_point), "landmark_id": landmark_id, "role": role, "review_status": review_status, "annotation_source": "cvat", "cvat_track_id": track.track_id, "proposal_metadata": dict(sorted(attributes.items()))}
                    if reviewed:
                        landmark.update({"reviewer": reviewer, "revision": int(revision), "reviewed_at": reviewed_at, "annotation_confidence": float(annotation_confidence)})
                    landmarks.append(landmark)
                else:
                    point_proposals.append({"shot_id": shot_id, "source_frame": mapped.source_frame, "pts": mapped.source_pts, "label": track.label, "image_xy_px": list(source_point), "review_status": "unreviewed", "attributes": dict(sorted(attributes.items()))})
            continue
        for box in track.boxes:
            if box.task_frame not in source_by_task:
                raise CvatError(f"CVAT shape references unmapped task frame {box.task_frame}")
            mapped = source_by_task[box.task_frame]
            if not start_frame <= mapped.source_frame < end_frame:
                raise CvatError(f"CVAT shape frame {mapped.source_frame} lies outside declared shot {shot_id}")
            # An outside box is a track-state marker with no visible box to
            # score. It is not an annotation and needs no review status.
            if box.outside:
                continue
            attributes = {**track.attributes, **(box.attributes or {})}
            if attributes.get("source_sha256") not in (None, "", frame_map.source_sha256):
                raise CvatError(f"CVAT track {track.track_id} source hash does not match task frame map")
            for key, expected in (("source_frame", mapped.source_frame), ("source_pts", mapped.source_pts)):
                # The frame map is authoritative for interpolated shapes;
                # CVAT can copy keyframe attributes forward unchanged.
                if box.keyframe and attributes.get(key) not in (None, ""):
                    try:
                        actual = int(attributes[key])
                    except (TypeError, ValueError) as error:
                        raise CvatError(f"invalid {key} attribute on track {track.track_id}") from error
                    if actual != expected:
                        raise CvatError(f"CVAT {key} attribute disagrees with task frame map at task frame {box.task_frame}")
            review_status = _import_review_status(attributes, reviewed=reviewed, label=track.label, frame=mapped.source_frame)
            if review_status is None:
                continue
            record: dict[str, Any] = {
                "id": f"{shot_id}:{tracklet_id}:{mapped.source_frame}",
                "shot_id": shot_id,
                "source_shot_id": str(attributes.get("source_shot_id") or shot_id),
                "source_frame": mapped.source_frame,
                "pts": mapped.source_pts,
                "track_id": tracklet_id,
                "label": track.label,
                "review_status": review_status,
                "coordinate_space": "source",
                "visibility": "occluded" if box.occluded else "visible",
                "annotation_source": "cvat",
                "cvat_track_id": track.track_id,
                "cvat_track_source": track.source,
                "proposal_metadata": dict(sorted(attributes.items())),
            }
            raw_inference_json = attributes.get("source_inference_row_json")
            if raw_inference_json not in (None, ""):
                try:
                    raw_inference = json.loads(raw_inference_json)
                except json.JSONDecodeError as error:
                    raise CvatError(f"invalid source inference provenance on CVAT track {track.track_id}") from error
                if not isinstance(raw_inference, dict):
                    raise CvatError(f"source inference provenance on CVAT track {track.track_id} must be an object")
                source_run_id = str(attributes.get("source_run_id", "")).strip()
                source_shot_id = str(attributes.get("source_shot_id", "")).strip()
                if not source_run_id or not source_shot_id:
                    raise CvatError("source inference provenance requires source_run_id and source_shot_id")
                for key, expected in (("run_id", source_run_id), ("shot_id", source_shot_id), ("tracklet_id", tracklet_id)):
                    if str(raw_inference.get(key, "")) != expected:
                        raise CvatError(f"source inference {key} does not match CVAT track {track.track_id}")
                try:
                    inference_frame = int(raw_inference["frame_index"])
                    inference_pts = int(raw_inference["pts"])
                except (KeyError, TypeError, ValueError) as error:
                    if box.keyframe:
                        raise CvatError(f"source inference row on CVAT track {track.track_id} has invalid frame_index or pts") from error
                    inference_frame = inference_pts = None
                if (inference_frame, inference_pts) == (mapped.source_frame, mapped.source_pts):
                    record["inference_provenance"] = {"run_id": source_run_id, "raw_row": raw_inference}
                elif box.keyframe:
                    raise CvatError(f"source inference frame/PTS does not match CVAT task frame {box.task_frame}")
            if box.bbox_xyxy_px is not None:
                source_box = task_bbox_to_source(box.bbox_xyxy_px, mapped)
                if source_box[0] < 0 or source_box[1] < 0 or source_box[2] > frame_map.source_width or source_box[3] > frame_map.source_height:
                    raise CvatError(f"imported box lies outside source image at frame {mapped.source_frame}")
                record["bbox_xyxy_px"] = list(source_box)
            for output_key, attribute_key in (("team", "team"), ("global_id", "global_id"), ("jersey_number", "jersey_number")):
                value = attributes.get(attribute_key)
                if value not in (None, "", "unknown", "ambiguous", "unresolved"):
                    record[output_key] = int(value) if output_key == "jersey_number" else value
            if attributes.get("team_suggestion") not in (None, ""):
                record["team_suggestion"] = str(attributes["team_suggestion"])
            if attributes.get("cross_shot_review_status") not in (None, ""):
                record["cross_shot_review_status"] = str(attributes["cross_shot_review_status"])
            for key in ("identity_second_reviewer", "identity_second_reviewed_at"):
                if attributes.get(key) not in (None, ""):
                    record[key] = str(attributes[key])
            if attributes.get("identity_second_revision") not in (None, ""):
                try:
                    record["identity_second_revision"] = int(attributes["identity_second_revision"])
                except ValueError as error:
                    raise CvatError(f"invalid identity_second_revision on CVAT track {track.track_id}") from error
            if attributes.get("identity_second_confidence") not in (None, ""):
                try:
                    record["identity_second_confidence"] = float(attributes["identity_second_confidence"])
                except ValueError as error:
                    raise CvatError(f"invalid identity_second_confidence on CVAT track {track.track_id}") from error
            for output_key, attribute_key in (("detection_score", "detection_score"), ("team_score", "team_score")):
                if attributes.get(attribute_key) not in (None, ""):
                    try:
                        score = float(attributes[attribute_key])
                    except (TypeError, ValueError) as error:
                        raise CvatError(f"invalid {attribute_key} attribute on track {track.track_id}") from error
                    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                        raise CvatError(f"{attribute_key} attribute must be between zero and one")
                    record[output_key] = score
            for output_key, attribute_key in (("ground_contact_confidence", "ground_contact_confidence"),):
                if attributes.get(attribute_key) not in (None, ""):
                    try:
                        confidence = float(attributes[attribute_key])
                    except ValueError as error:
                        raise CvatError(f"invalid {attribute_key} attribute on track {track.track_id}") from error
                    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                        raise CvatError(f"{attribute_key} attribute must be between zero and one")
                    record[output_key] = confidence
            if attributes.get("ground_contact_xy_yards"):
                try:
                    contact = json.loads(attributes["ground_contact_xy_yards"])
                    if not isinstance(contact, list) or len(contact) != 2:
                        raise ValueError("expected two coordinates")
                    record["ground_contact_xy_yards"] = [float(item) for item in contact]
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise CvatError(f"invalid ground_contact_xy_yards on track {track.track_id}") from error
            if reviewed:
                record.update({"reviewer": reviewer, "revision": int(revision), "reviewed_at": reviewed_at, "annotation_confidence": float(annotation_confidence)})
            records.append(record)
    records_by_key = {(record.get("track_id"), record.get("source_frame")): record for record in records}
    for (tracklet_id, source_frame), (point, confidence, attributes, review_status) in sorted(contact_points.items()):
        mapped = source_by_frame.get(source_frame)
        if mapped is None:
            raise CvatError(f"ground-contact source frame {source_frame} is absent from task map")
        record = records_by_key.get((tracklet_id, source_frame))
        if record is None:
            record = {"id": f"{shot_id}:{tracklet_id}:{source_frame}", "shot_id": shot_id, "source_frame": source_frame, "pts": mapped.source_pts, "track_id": tracklet_id, "label": "player", "review_status": review_status, "coordinate_space": "source", "annotation_source": "cvat_ground_contact", "cvat_point_attributes": attributes}
            records.append(record)
            records_by_key[(tracklet_id, source_frame)] = record
        record["ground_contact_xy_px"] = list(point)
        if confidence is not None:
            record["ground_contact_confidence"] = confidence
        if reviewed:
            record.update({"reviewer": reviewer, "revision": int(revision), "reviewed_at": reviewed_at, "annotation_confidence": float(annotation_confidence)})
    frame_labels_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for tag in annotations.tags:
        if tag.task_frame not in source_by_task:
            raise CvatError(f"CVAT tag references unmapped task frame {tag.task_frame}")
        mapped = source_by_task[tag.task_frame]
        if not start_frame <= mapped.source_frame < end_frame:
            raise CvatError(f"CVAT tag frame {mapped.source_frame} lies outside declared shot {shot_id}")
        attributes = tag.attributes
        review_status = _import_review_status(attributes, reviewed=reviewed, label=tag.label, frame=mapped.source_frame)
        if review_status is None:
            continue
        if tag.label not in {"reviewed_frame", "empty_frame", "ignored_frame"} and attributes.get("labeled") not in {"true", "1"}:
            continue
        frame_key = (shot_id, mapped.source_frame)
        frame_label: dict[str, Any] = frame_labels_by_key.setdefault(frame_key, {
            "shot_id": shot_id,
            "source_frame": mapped.source_frame,
            "pts": mapped.source_pts,
            "labeled": False,
            "ignore": False,
            "review_status": review_status,
        })
        frame_label["labeled"] = bool(frame_label["labeled"] or attributes.get("labeled", "true") in {"true", "1"} or tag.label in {"reviewed_frame", "empty_frame", "ignored_frame"})
        frame_label["ignore"] = bool(frame_label["ignore"] or tag.label == "ignored_frame" or attributes.get("ignore", "false") in {"true", "1"})
    for frame_label in frame_labels_by_key.values():
        if reviewed:
            frame_label.update({"reviewer": reviewer, "revision": int(revision), "reviewed_at": reviewed_at, "annotation_confidence": float(annotation_confidence)})
        frame_labels.append(frame_label)
    return records, landmarks, frame_labels, point_proposals, timing_events
