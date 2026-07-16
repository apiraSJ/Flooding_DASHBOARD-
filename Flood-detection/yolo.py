"""
YOLO Flood Detection Video Processor
====================================
Processes video files using YOLOv8 for flood detection and water level measurement.
Detects all valid flood areas in video frames and calculates distance from water to
a reference line without crashing on empty detections or invalid geometry.
"""

import math
import os
import threading
import requests

import cv2
import numpy
import torch.serialization
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, Point, Polygon
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel


CONF_THRESHOLD = 0.5
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
OUTPUT_VIDEO_PATH = "Output Video.mp4"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
REPORT_URL = os.environ.get("REPORT_URL", "http://localhost:5000/api/report_flood")

# Maps a compass label to the bearing convention used by server.py's report_flood
# endpoint, so a caller who knows which way the camera faces can pass it through.
VALID_DIRECTIONS = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}


def send_report_async(payload, on_error=None):
    def target():
        try:
            requests.post(REPORT_URL, json=payload, timeout=2)
        except Exception as exc:
            # Previously this swallowed every failure silently, so a misconfigured
            # REPORT_URL or a downed server meant detections vanished with no trace.
            if on_error:
                on_error(exc)
            else:
                print(f"[yolo] failed to send report to {REPORT_URL}: {exc}")
    threading.Thread(target=target, daemon=True).start()

MODEL = None


def get_model():
    """Load YOLO once and reuse it across calls."""
    global MODEL
    if MODEL is None:
        torch.serialization.add_safe_globals([SegmentationModel])
        MODEL = YOLO(MODEL_PATH)
        MODEL.to("cpu")
    return MODEL


def load_font(size=45):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def validate_video_input(video_path, pixels_in_a_meter):
    if pixels_in_a_meter <= 0:
        raise ValueError("Invalid pixel scale: pixelsInAMeter must be greater than 0.")

    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format: {ext}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise ValueError("Invalid video input: FPS is 0.")
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video input: empty frame size.")
    if frame_count <= 0:
        cap.release()
        raise ValueError("Invalid video input: no readable frames.")

    return cap, fps, width, height


def calculate_distance(x1, y1, x2, y2, pixels_in_a_meter, tip_height):
    distance = tip_height - (math.hypot(x2 - x1, y2 - y1) / pixels_in_a_meter)
    return float(f"{distance:.2f}")


def get_detection_confidences(result):
    boxes = getattr(result, "boxes", None)
    conf = getattr(boxes, "conf", None)
    if conf is None:
        return []
    return [float(value) for value in conf.tolist()]


def get_mask_polygons(result, frame_shape):
    masks = getattr(result, "masks", None)
    if masks is None:
        return []

    height, width = frame_shape[:2]
    confidences = get_detection_confidences(result)
    polygons = []

    if getattr(masks, "xy", None) is not None:
        raw_segments = masks.xy
        normalized = False
    elif getattr(masks, "xyn", None) is not None:
        raw_segments = masks.xyn
        normalized = True
    elif getattr(masks, "segments", None) is not None:
        raw_segments = masks.segments
        normalized = True
    else:
        return []

    for index, segment in enumerate(raw_segments):
        confidence = confidences[index] if index < len(confidences) else 1.0
        if confidence < CONF_THRESHOLD:
            continue

        vertices = []
        for point in segment:
            if len(point) < 2:
                continue
            x = float(point[0])
            y = float(point[1])
            if normalized:
                x *= width
                y *= height
            vertices.append((int(round(x)), int(round(y))))

        if len(vertices) < 3:
            continue

        polygon = Polygon(vertices)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            continue

        polygons.append({
            "polygon": polygon,
            "vertices": vertices,
            "confidence": confidence,
            "bbox": polygon.bounds,
        })

    return polygons


def intersection_points(geometry):
    if geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [(geometry.x, geometry.y)]
    if isinstance(geometry, MultiPoint):
        return [(point.x, point.y) for point in geometry.geoms]
    if isinstance(geometry, LineString):
        return list(geometry.coords)
    if isinstance(geometry, MultiLineString):
        points = []
        for line in geometry.geoms:
            points.extend(list(line.coords))
        return points
    if isinstance(geometry, GeometryCollection):
        points = []
        for item in geometry.geoms:
            points.extend(intersection_points(item))
        return points
    return []


def nearest_intersection(annotation, line, start_x, start_y):
    intersection = annotation.intersection(line)
    points = intersection_points(intersection)
    if not points:
        return None
    return min(points, key=lambda point: math.hypot(point[0] - start_x, point[1] - start_y))


def draw_status(pil_image, font, label, fill):
    draw = ImageDraw.Draw(pil_image)
    draw.text((936, 98), label, font=font, fill=fill)


def yolo(video_path, firstCoordinate_x, firstCoordinate_y, secondCoordinate_x, secondCoordinate_y, pixelsInAMeter, tipHeight, warningLevel, lat=13.7563, lng=100.5018, user_dir="N", show_preview=False):
    """
    Process video for flood detection using YOLOv8 and water level measurement.

    Outputs an annotated video to 'Output Video.mp4'. The function keeps the
    original public signature for compatibility and returns a processing summary.

    lat/lng: fixed GPS position of the camera/reference line being monitored.
    user_dir: compass bearing ("N","NE","E","SE","S","SW","W","NW") describing which
        way the survivor is from the camera, forwarded to the server so the placeholder
        survivor marker isn't always pinned directly north of the hazard by default.
    show_preview: set True to pop up a live cv2.imshow() window. Leave False (default)
        when running headless (e.g. alongside the Flask server on a machine with no
        display attached) — imshow()/waitKey() raise/hang without a display server.
    """
    if user_dir not in VALID_DIRECTIONS:
        raise ValueError(f"user_dir must be one of {sorted(VALID_DIRECTIONS)}, got {user_dir!r}")

    font = load_font()
    distances = []
    detections_processed = 0
    frames_processed = 0
    last_sent_status = None

    cap, fps, width, height = validate_video_input(video_path, pixelsInAMeter)
    output_video = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not output_video.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {OUTPUT_VIDEO_PATH}")

    model = get_model()
    line = LineString([(firstCoordinate_x, firstCoordinate_y), (secondCoordinate_x, secondCoordinate_y)])

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frames_processed += 1
            results = model(frame, conf=CONF_THRESHOLD)
            result = results[0]
            polygons = get_mask_polygons(result, frame.shape)

            annotated_frame = result.plot() if polygons else frame.copy()
            color_converted = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_converted)
            draw = ImageDraw.Draw(pil_image)

            frame_distances = []
            frame_confidences = []
            for detection in polygons:
                point = nearest_intersection(
                    detection["polygon"],
                    line,
                    firstCoordinate_x,
                    firstCoordinate_y,
                )
                if point is None:
                    continue

                detections_processed += 1
                intersection_x, intersection_y = point
                distance = calculate_distance(
                    intersection_x,
                    intersection_y,
                    firstCoordinate_x,
                    firstCoordinate_y,
                    pixelsInAMeter,
                    tipHeight,
                )
                frame_distances.append(distance)
                frame_confidences.append(detection["confidence"])
                distances.append(distance)

                updated_line = LineString([
                    (firstCoordinate_x, firstCoordinate_y),
                    (intersection_x, intersection_y),
                ])
                draw.line(updated_line.coords, fill=(0, 255, 0), width=3)

            max_distance = 0.0
            max_confidence = 0.0
            if frame_distances:
                max_idx = max(range(len(frame_distances)), key=lambda i: frame_distances[i])
                max_distance = frame_distances[max_idx]
                max_confidence = frame_confidences[max_idx]
                draw.text((1013, 134), str(max_distance), font=font, fill=(0, 0, 0))
                if max_distance >= warningLevel:
                    draw.text((936, 98), "WARNING!!!", font=font, fill=(255, 0, 0))
                    status_str = "WARNING!!!"
                    severity = 8
                else:
                    draw.text((936, 98), "SAFE", font=font, fill=(0, 255, 0))
                    status_str = "SAFE"
                    severity = 4
            else:
                draw_status(pil_image, font, "NO FLOOD", (0, 255, 0))
                status_str = "NO FLOOD"
                severity = 0

            # Send periodic/state-change reports to the server
            if (frames_processed == 1) or (status_str != last_sent_status) or (frames_processed % 150 == 0):
                last_sent_status = status_str
                payload = {
                    "severity": severity,
                    "lat": lat,
                    "lng": lng,
                    # Real detection confidence for this frame's worst-case flood
                    # measurement, instead of always reporting the fixed threshold.
                    "confidence": round(max_confidence * 100),
                    "level_label": f"YOLO Video ({status_str})",
                    # How far the water's measured edge is from the reference line, and
                    # which way the survivor is from the camera. Previously these were
                    # never sent, so the server always placed the survivor at a fixed
                    # offset due north of the hazard regardless of what was measured.
                    "user_dist_m": max_distance,
                    "user_dir": user_dir,
                }
                send_report_async(payload)

            result_frame = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
            output_video.write(result_frame)

            if show_preview:
                try:
                    cv2.imshow("YOLOv8 Inference", result_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    # No display server available (e.g. running headless alongside
                    # the Flask backend) — keep processing the video without a preview
                    # window instead of crashing.
                    show_preview = False
    finally:
        output_video.release()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    return {
        "frames_processed": frames_processed,
        "detections_processed": detections_processed,
        "distances": distances,
        "output_video": OUTPUT_VIDEO_PATH,
    }
