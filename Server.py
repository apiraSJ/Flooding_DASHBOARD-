"""
RescuOpt AI — Flask Backend Server (Clean)
============================================
รับผลจาก YOLO detection → relay ข้อมูลให้ dashboard frontend
Algorithms run in browser (disaster_nav.html), not here.
"""

try:
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
except ImportError as e:
    raise RuntimeError(
        "Missing dependency: install Flask and Flask-CORS with 'pip install flask flask-cors'"
    ) from e

import math, os, json, threading
from pathlib import Path

# NOTE (security): the app no longer serves "." as a static folder — that exposed this
# very file, the YOLO model weights, and anything else in the project directory over
# HTTP. Only the "dashboard" subfolder (the actual frontend assets) is served.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

app = Flask(__name__, static_folder=None)

# CORS origins are configurable via env var instead of allowing every origin by default.
# For local development with the file served from the same Flask app this can stay "*",
# but if you expose this server beyond localhost, set CORS_ORIGINS to your dashboard's
# real origin(s), e.g. CORS_ORIGINS="https://your-dashboard.example.com"
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
CORS(app, origins=CORS_ORIGINS)

# ── Global storage ──────────────────────────────────────
# All mutable global state is guarded by STATE_LOCK. Flask's dev server (and most
# production WSGI servers) can process requests concurrently across threads, so without
# a lock, two requests touching GLOBAL_HAZARDS at the same time (e.g. /api/clear racing
# with /api/report_flood) could interleave and corrupt the list.
STATE_LOCK = threading.Lock()
GLOBAL_HAZARDS = []
GLOBAL_SURVIVOR = None
GLOBAL_EXIT = None
EARTH_RADIUS_M = 6371000
DEFAULT_HAZARD_RADIUS_M = 500
DEFAULT_SAFETY_MARGIN_M = 500
HAZARD_MERGE_TOL_M = 60  # hazards within this many meters of each other are treated as the same one (update, not duplicate)

# ── Coordinate system ──────────────────────────────────
REF_LAT = 13.7563
REF_LNG = 100.5018
DEG_M_LAT = 111320

def to_meters(lat, lng):
    dx = (lng - REF_LNG) * DEG_M_LAT * math.cos(math.radians(REF_LAT))
    dy = (lat - REF_LAT) * DEG_M_LAT
    return {"x": round(dx), "y": round(dy)}

def to_latlng(x, y):
    lng = REF_LNG + x / (DEG_M_LAT * math.cos(math.radians(REF_LAT)))
    lat = REF_LAT + y / DEG_M_LAT
    return [lat, lng]

def haversine(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def offset_coordinate(lat, lng, distance_m, bearing_deg):
    bearing = math.radians(bearing_deg)
    angular_distance = distance_m / EARTH_RADIUS_M
    lat1 = math.radians(lat)
    lng1 = math.radians(lng)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lng2 = lng1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    return {"lat": math.degrees(lat2), "lng": math.degrees(lng2)}

def total_danger(mx, my, hazards):
    total = 0.0
    for h in hazards:
        hm = to_meters(h["lat"], h["lng"])
        dist = math.hypot(mx - hm["x"], my - hm["y"])
        radius = h.get("radius_m", 200)
        if dist < radius:
            total += (h.get("severity", 5) / 10.0) * (1.0 - dist / radius)
    return min(total, 1.0)

def exit_is_safe(exit_point, hazards, safety_margin_m=DEFAULT_SAFETY_MARGIN_M):
    return all(
        haversine(exit_point["lat"], exit_point["lng"], h["lat"], h["lng"])
        > (h.get("radius_m") or DEFAULT_HAZARD_RADIUS_M) + safety_margin_m
        for h in hazards
    )

def generate_safe_exit(survivor, hazards, safety_margin_m=DEFAULT_SAFETY_MARGIN_M):
    if not survivor:
        return None
    if not hazards:
        return offset_coordinate(survivor["lat"], survivor["lng"], 5000, 0)
    nearest = min(
        hazards,
        key=lambda h: haversine(survivor["lat"], survivor["lng"], h["lat"], h["lng"]),
    )
    nearest_distance = max(
        haversine(survivor["lat"], survivor["lng"], nearest["lat"], nearest["lng"]), 1.0,
    )
    dlat = math.radians(survivor["lat"] - nearest["lat"])
    dlng = math.radians(survivor["lng"] - nearest["lng"])
    y = math.sin(dlng) * math.cos(math.radians(survivor["lat"]))
    x = (
        math.cos(math.radians(nearest["lat"])) * math.sin(math.radians(survivor["lat"]))
        - math.sin(math.radians(nearest["lat"])) * math.cos(math.radians(survivor["lat"])) * math.cos(dlng)
    )
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360 if abs(dlat) + abs(dlng) > 1e-12 else 0
    minimum_distance = (nearest.get("radius_m") or DEFAULT_HAZARD_RADIUS_M) + safety_margin_m
    candidate_distance = max(nearest_distance + safety_margin_m, minimum_distance)
    for _ in range(60):
        candidate = offset_coordinate(nearest["lat"], nearest["lng"], candidate_distance, bearing)
        if exit_is_safe(candidate, hazards, safety_margin_m):
            return candidate
        candidate_distance += safety_margin_m
    return offset_coordinate(nearest["lat"], nearest["lng"], candidate_distance, bearing)

# ── Static file routes ────────────────────────────────
# Only DASHBOARD_DIR is ever handed to send_from_directory, so requests can't escape
# into the rest of the project (e.g. this server.py file or the YOLO model weights).
@app.route("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "disaster_nav.html")

@app.route("/disaster_nav")
def disaster_nav_redirect():
    return send_from_directory(DASHBOARD_DIR, "disaster_nav.html")

@app.route("/dashboard/<path:filename>")
def serve_dashboard(filename):
    return send_from_directory(DASHBOARD_DIR, filename)

@app.route("/api/hazards", methods=["GET"])
def get_hazards():
    with STATE_LOCK:
        return jsonify({
            "hazards": list(GLOBAL_HAZARDS),
            "survivor": GLOBAL_SURVIVOR,
            "exit": GLOBAL_EXIT,
        })

@app.route("/api/clear", methods=["POST"])
def clear_all():
    global GLOBAL_HAZARDS, GLOBAL_SURVIVOR, GLOBAL_EXIT
    with STATE_LOCK:
        GLOBAL_HAZARDS.clear()
        GLOBAL_SURVIVOR = None
        GLOBAL_EXIT = None
    return jsonify({"status": "cleared"})


@app.route("/api/remove_hazard", methods=["POST"])
def remove_hazard():
    global GLOBAL_HAZARDS
    data = request.json
    lat = data.get("lat")
    lng = data.get("lng")
    if lat is None or lng is None:
        return jsonify({"error": "lat/lng required"}), 400
    with STATE_LOCK:
        before = len(GLOBAL_HAZARDS)
        GLOBAL_HAZARDS = [
            h for h in GLOBAL_HAZARDS
            if not (abs(h.get("lat", 0) - lat) < 0.0001 and abs(h.get("lng", 0) - lng) < 0.0001)
        ]
        removed = before - len(GLOBAL_HAZARDS)
    return jsonify({"status": "ok", "removed": removed})


@app.route("/api/report_flood", methods=["POST"])
def report_flood():
    data = request.json or {}
    severity = data.get("severity", 5)
    confidence = data.get("confidence", 0)
    lat = data.get("lat", 13.7563)
    lng = data.get("lng", 100.5018)

    if severity <= 3:
        radius_m = 200 + int(confidence * 6)
    elif severity <= 6:
        radius_m = 500 + int(confidence * 20)
    else:
        radius_m = 1000 + int(confidence * 40)

    new_hazard = {
        "type": "flood",
        "lat": lat, "lng": lng,
        "severity": severity, "radius_m": radius_m,
        "confidence": data.get("confidence", 0),
        "level_label": data.get("level_label", "Unknown")
    }

    global GLOBAL_SURVIVOR, GLOBAL_EXIT
    with STATE_LOCK:
        if severity <= 0:
            # "NO FLOOD" report: don't draw a fake hazard circle on the map. Previously
            # every report (including "no flood") appended a new hazard, so a long video
            # with a stable camera would flood the map with hundreds of overlapping,
            # meaningless circles at severity 0.
            existing_idx = None
        else:
            # Update the existing hazard at (roughly) this location instead of appending
            # a duplicate every time YOLO re-reports the same physical spot. Without this,
            # a long-running video sends a report every ~150 frames or on every status
            # change, and GLOBAL_HAZARDS grows without bound at the same coordinates.
            existing_idx = None
            for i, h in enumerate(GLOBAL_HAZARDS):
                if haversine(lat, lng, h["lat"], h["lng"]) <= HAZARD_MERGE_TOL_M:
                    existing_idx = i
                    break

            if existing_idx is not None:
                GLOBAL_HAZARDS[existing_idx] = new_hazard
            else:
                GLOBAL_HAZARDS.append(new_hazard)

        user_dist_m = float(data.get("user_dist_m", 0))
        user_dir = data.get("user_dir", "N")
        bearings = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
        bearing_deg = bearings.get(user_dir, 0)

        total_dist_from_center_m = radius_m + user_dist_m
        survivor = offset_coordinate(lat, lng, total_dist_from_center_m, bearing_deg)

        GLOBAL_SURVIVOR = survivor
        GLOBAL_EXIT = generate_safe_exit(survivor, GLOBAL_HAZARDS)

    return jsonify({"status": "received", "hazard": new_hazard})

@app.route("/api/report_flood_v2", methods=["POST"])
def report_flood_v2():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    survivors = data.get("survivors", [])
    hazards = data.get("hazards", [])
    severity = data.get("severity", 5)
    confidence = data.get("confidence", 0)
    level_label = data.get("level_label", "Unknown")

    global GLOBAL_SURVIVOR, GLOBAL_EXIT
    with STATE_LOCK:
        GLOBAL_HAZARDS.clear()
        seen_hazards = set()

        for h in hazards:
            lat = h.get("lat", 13.7563)
            lng = h.get("lng", 100.5018)
            radius_m = h.get("radius_m") or DEFAULT_HAZARD_RADIUS_M
            hazard_key = (round(lat, 7), round(lng, 7), h.get("type", "flood"))
            if hazard_key in seen_hazards:
                continue
            seen_hazards.add(hazard_key)
            GLOBAL_HAZARDS.append({
                "type": h.get("type", "flood"),
                "lat": lat, "lng": lng,
                "severity": h.get("severity", severity),
                "radius_m": radius_m,
                "confidence": confidence,
                "level_label": level_label,
            })

        if survivors:
            first = survivors[0]
            GLOBAL_SURVIVOR = {"lat": first["lat"], "lng": first["lng"]}
            GLOBAL_EXIT = generate_safe_exit(GLOBAL_SURVIVOR, GLOBAL_HAZARDS)

    return jsonify({
        "status": "received",
        "survivors_count": len(survivors),
        "hazards_count": len(GLOBAL_HAZARDS),
        "exit": GLOBAL_EXIT,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Debug mode defaults to OFF: Werkzeug's interactive debugger lets anyone who can
    # reach an endpoint that raises an exception execute arbitrary code on the server.
    # Only enable it locally with FLASK_DEBUG=1 while developing.
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"RescuOpt AI Server starting on http://127.0.0.1:{port}/dashboard/disaster_nav.html")
    app.run(debug=debug, port=port)
