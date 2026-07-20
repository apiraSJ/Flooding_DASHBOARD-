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
    """แปลงพิกัด GPS (lat, lng) ให้เป็นระยะทางหน่วยเมตร (x, y) เทียบกับจุดอ้างอิง REF_LAT/REF_LNG

    ใช้ flat-earth approximation (ไม่ใช่ great-circle) ซึ่งแม่นยำพอสำหรับพื้นที่ขนาดเล็ก
    (ระดับเมือง/จังหวัด) และช่วยให้อัลกอริทึมหาเส้นทางฝั่งเว็บคำนวณบนกริดเมตรธรรมดาได้ง่ายกว่า
    การคำนวณบนพิกัดองศาโดยตรง

    Args:
        lat (float): ละติจูด (องศา)
        lng (float): ลองจิจูด (องศา)

    Returns:
        dict: {"x": ระยะทางแกน East-West เป็นเมตร, "y": ระยะทางแกน North-South เป็นเมตร}
        ทั้งคู่ปัดเป็นจำนวนเต็มด้วย round()
    """
    dx = (lng - REF_LNG) * DEG_M_LAT * math.cos(math.radians(REF_LAT))
    dy = (lat - REF_LAT) * DEG_M_LAT
    return {"x": round(dx), "y": round(dy)}

def to_latlng(x, y):
    """แปลงระยะทางหน่วยเมตร (x, y) กลับเป็นพิกัด GPS (lat, lng)

    เป็นฟังก์ชันผกผัน (inverse) ของ to_meters() ใช้แปลงผลลัพธ์จากการคำนวณบนกริดเมตร
    (เช่น เส้นทางที่อัลกอริทึมหาได้) กลับไปเป็นพิกัดจริงเพื่อส่งให้ frontend วาดบนแผนที่

    Args:
        x (float): ระยะทางแกน East-West เป็นเมตร (เทียบกับจุดอ้างอิง)
        y (float): ระยะทางแกน North-South เป็นเมตร (เทียบกับจุดอ้างอิง)

    Returns:
        list[float, float]: [lat, lng] พิกัด GPS ที่แปลงกลับมา
    """
    lng = REF_LNG + x / (DEG_M_LAT * math.cos(math.radians(REF_LAT)))
    lat = REF_LAT + y / DEG_M_LAT
    return [lat, lng]

def haversine(lat1, lng1, lat2, lng2):
    """คำนวณระยะทางจริงบนผิวโลก (เมตร) ระหว่างพิกัด GPS สองจุด ด้วยสูตร Haversine

    ต่างจาก to_meters()/to_latlng() ตรงที่ฟังก์ชันนี้แม่นยำกว่าสำหรับการวัดระยะทางตรง ๆ
    (great-circle distance) จึงถูกใช้ในจุดที่ต้องเทียบระยะปลอดภัย เช่น ตรวจว่าจุดอพยพ
    ห่างจากจุดอันตรายพอหรือยัง

    Args:
        lat1, lng1 (float): พิกัดจุดที่ 1 (องศา)
        lat2, lng2 (float): พิกัดจุดที่ 2 (องศา)

    Returns:
        float: ระยะทางเป็นเมตรระหว่างสองจุด
    """
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def offset_coordinate(lat, lng, distance_m, bearing_deg):
    """หาพิกัด GPS ใหม่ที่อยู่ห่างจากจุดเริ่มต้นไปในทิศทางและระยะทางที่กำหนด

    ใช้สูตรคำนวณพิกัดปลายทางบนทรงกลม (spherical destination formula) เพื่อ "เดิน" จาก
    จุดหนึ่งไปตามทิศทาง (bearing) เป็นระยะทางที่ต้องการ แล้วได้พิกัดปลายทางกลับมา
    ใช้ในการวาง survivor แบบ placeholder และการขยับหาจุดอพยพที่ปลอดภัย

    Args:
        lat, lng (float): พิกัดจุดเริ่มต้น (องศา)
        distance_m (float): ระยะทางที่ต้องการเดินจากจุดเริ่มต้น (เมตร)
        bearing_deg (float): ทิศทางเป็นมุมองศา (0 = เหนือ, 90 = ตะวันออก, ตามเข็มนาฬิกา)

    Returns:
        dict: {"lat": ละติจูดใหม่, "lng": ลองจิจูดใหม่}
    """
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
    """คำนวณ "คะแนนความอันตราย" รวม (0.0–1.0) ของจุด (mx, my) บนกริดเมตร

    สำหรับจุดอันตรายแต่ละจุดที่จุด (mx, my) อยู่ในรัศมี จะบวกคะแนนตามสัดส่วนความรุนแรง
    (severity/10) คูณด้วยความใกล้ศูนย์กลาง (1 - dist/radius) — ยิ่งใกล้จุดอันตรายและ
    ยิ่งรุนแรงมาก คะแนนยิ่งสูง ผลรวมทั้งหมดถูก clamp ไว้ไม่เกิน 1.0 ใช้เป็นค่าอ้างอิง
    ความปลอดภัยของพื้นที่ (ค่าเดียวกับที่ dashboard ฝั่งเว็บใช้ในอัลกอริทึมหาเส้นทาง)

    Args:
        mx, my (float): ตำแหน่งที่ต้องการประเมิน บนกริดเมตร (มาจาก to_meters())
        hazards (list[dict]): รายการจุดอันตราย แต่ละจุดมี lat, lng, radius_m, severity

    Returns:
        float: คะแนนความอันตรายรวม อยู่ในช่วง 0.0 (ปลอดภัยสนิท) ถึง 1.0 (อันตรายสูงสุด)
    """
    total = 0.0
    for h in hazards:
        hm = to_meters(h["lat"], h["lng"])
        dist = math.hypot(mx - hm["x"], my - hm["y"])
        radius = h.get("radius_m", 200)
        if dist < radius:
            total += (h.get("severity", 5) / 10.0) * (1.0 - dist / radius)
    return min(total, 1.0)

def exit_is_safe(exit_point, hazards, safety_margin_m=DEFAULT_SAFETY_MARGIN_M):
    """ตรวจสอบว่าจุด exit_point ห่างจากจุดอันตราย "ทุกจุด" มากกว่า (รัศมีอันตราย + ระยะกันชน) หรือไม่

    ใช้เป็นเงื่อนไขยืนยันความปลอดภัยก่อนเลือกจุดใดจุดหนึ่งเป็น "จุดอพยพ" (safe exit)
    ต้องผ่านเงื่อนไขนี้กับจุดอันตรายทุกจุดพร้อมกัน (all()) จึงจะถือว่าปลอดภัย

    Args:
        exit_point (dict): {"lat": ..., "lng": ...} จุดที่ต้องการตรวจสอบ
        hazards (list[dict]): รายการจุดอันตรายทั้งหมด
        safety_margin_m (float): ระยะกันชนเพิ่มเติมนอกเหนือจากรัศมีอันตราย (เมตร)

    Returns:
        bool: True ถ้าปลอดภัยจากจุดอันตรายทุกจุด, False ถ้าอยู่ใกล้จุดใดจุดหนึ่งเกินไป
    """
    return all(
        haversine(exit_point["lat"], exit_point["lng"], h["lat"], h["lng"])
        > (h.get("radius_m") or DEFAULT_HAZARD_RADIUS_M) + safety_margin_m
        for h in hazards
    )

def generate_safe_exit(survivor, hazards, safety_margin_m=DEFAULT_SAFETY_MARGIN_M):
    """คำนวณหาจุดอพยพ (exit point) ที่ปลอดภัยที่สุดโดยอัตโนมัติ ให้ผู้รอดชีวิต 1 คน

    วิธีการ:
        1. ถ้าไม่มี survivor → คืนค่า None (ไม่มีอะไรให้คำนวณ)
        2. ถ้าไม่มี hazard เลย → เดินออกจากตำแหน่ง survivor ไป 5000 เมตร (ทิศเหนือ) ถือว่าปลอดภัย
        3. ถ้ามี hazard → หาจุดอันตรายที่ใกล้ survivor ที่สุด แล้วคำนวณทิศทาง (bearing)
           จากจุดอันตรายนั้นไปยัง survivor เพื่อใช้เป็นทิศ "หนีออก" จากศูนย์กลางอันตราย
        4. เดินออกจากจุดอันตรายนั้นไปตามทิศทางดังกล่าวทีละ safety_margin_m จนกว่า
           exit_is_safe() จะยืนยันว่าจุดนั้นปลอดภัยจากจุดอันตรายทุกจุด (ลองสูงสุด 60 รอบ)

    Args:
        survivor (dict | None): {"lat": ..., "lng": ...} ตำแหน่งผู้รอดชีวิต
        hazards (list[dict]): รายการจุดอันตรายทั้งหมด
        safety_margin_m (float): ระยะกันชนที่ต้องการให้ห่างจากจุดอันตรายเพิ่มเติม (เมตร)

    Returns:
        dict | None: {"lat": ..., "lng": ...} จุดอพยพที่ปลอดภัย หรือ None ถ้าไม่มี survivor
    """
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
    """หน้าแรกของเซิร์ฟเวอร์ (root "/") — ส่งไฟล์ dashboard หลัก (disaster_nav.html) กลับไป

    Returns:
        Response: ไฟล์ HTML ของ dashboard จาก DASHBOARD_DIR
    """
    return send_from_directory(DASHBOARD_DIR, "disaster_nav.html")

@app.route("/disaster_nav")
def disaster_nav_redirect():
    """เส้นทางสำรอง "/disaster_nav" — ส่งไฟล์ dashboard หลักกลับไปเช่นเดียวกับ index()

    Returns:
        Response: ไฟล์ HTML ของ dashboard จาก DASHBOARD_DIR
    """
    return send_from_directory(DASHBOARD_DIR, "disaster_nav.html")

@app.route("/dashboard/<path:filename>")
def serve_dashboard(filename):
    """เสิร์ฟไฟล์ static ใดๆ ภายในโฟลเดอร์ dashboard (เช่น css, js, image)

    Args:
        filename (str): ชื่อไฟล์ (หรือ path ย่อย) ที่ต้องการภายใน DASHBOARD_DIR

    Returns:
        Response: ไฟล์ที่ร้องขอจาก DASHBOARD_DIR
    """
    return send_from_directory(DASHBOARD_DIR, filename)

@app.route("/api/hazards", methods=["GET"])
def get_hazards():
    """คืนสถานะปัจจุบันของ hazard/survivor/exit ทั้งหมดที่เซิร์ฟเวอร์เก็บไว้ (global state)

    Returns:
        Response: JSON ที่มีคีย์ "hazards" (list), "survivor" (dict หรือ None),
        และ "exit" (dict หรือ None)
    """
    with STATE_LOCK:
        return jsonify({
            "hazards": list(GLOBAL_HAZARDS),
            "survivor": GLOBAL_SURVIVOR,
            "exit": GLOBAL_EXIT,
        })

@app.route("/api/clear", methods=["POST"])
def clear_all():
    """ล้างสถานะทั้งหมด (hazards, survivor, exit) กลับเป็นค่าว่าง

    Returns:
        Response: JSON {"status": "cleared"}
    """
    global GLOBAL_HAZARDS, GLOBAL_SURVIVOR, GLOBAL_EXIT
    with STATE_LOCK:
        GLOBAL_HAZARDS.clear()
        GLOBAL_SURVIVOR = None
        GLOBAL_EXIT = None
    return jsonify({"status": "cleared"})


@app.route("/api/remove_hazard", methods=["POST"])
def remove_hazard():
    """ลบ hazard ที่ตรงกับพิกัด lat/lng ที่ระบุ (ในระยะ tolerance 0.0001 องศา) ออกจาก GLOBAL_HAZARDS

    Expects JSON body:
        lat (float): ละติจูดของ hazard ที่ต้องการลบ
        lng (float): ลองจิจูดของ hazard ที่ต้องการลบ

    Returns:
        Response: JSON {"status": "ok", "removed": <จำนวนที่ถูกลบ>}
        หรือ ({"error": "lat/lng required"}, 400) ถ้าไม่ส่ง lat/lng มา
    """
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
    """รับผลตรวจจับน้ำท่วมจาก YOLO (severity/confidence เดี่ยว) แล้วอัปเดต/สร้าง hazard,
    คำนวณตำแหน่ง survivor จากระยะและทิศทางที่ผู้ใช้ระบุ และหาทางออกที่ปลอดภัยให้อัตโนมัติ

    Expects JSON body:
        severity (int, optional): ระดับความรุนแรงของน้ำท่วม (default 5)
        confidence (float, optional): ความมั่นใจของโมเดล YOLO (default 0)
        lat (float, optional): ละติจูดจุดตรวจพบ (default 13.7563)
        lng (float, optional): ลองจิจูดจุดตรวจพบ (default 100.5018)
        user_dist_m (float, optional): ระยะห่างของผู้ใช้จากจุดตรวจพบ (เมตร)
        user_dir (str, optional): ทิศทางของผู้ใช้ (N/NE/E/SE/S/SW/W/NW)
        level_label (str, optional): ป้ายกำกับระดับความรุนแรง

    Returns:
        Response: JSON {"status": "received", "hazard": <hazard ที่สร้าง/อัปเดต>}
    """
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
    """รับผลตรวจจับน้ำท่วมแบบชุด (หลาย survivors/hazards พร้อมกัน) แทนที่สถานะ hazard
    ทั้งหมดด้วยข้อมูลชุดใหม่ กรอง hazard ซ้ำด้วยพิกัด+ประเภท และคำนวณ exit ที่ปลอดภัยให้ survivor แรก

    Expects JSON body:
        survivors (list, optional): รายการผู้รอดชีวิต แต่ละรายการมี lat/lng
        hazards (list, optional): รายการ hazard แต่ละรายการมี lat/lng/type/severity/radius_m
        severity (int, optional): ความรุนแรง default ถ้า hazard ไม่ได้ระบุ (default 5)
        confidence (float, optional): ความมั่นใจของโมเดล (default 0)
        level_label (str, optional): ป้ายกำกับระดับความรุนแรง (default "Unknown")

    Returns:
        Response: JSON {"status": "received", "survivors_count": int,
        "hazards_count": int, "exit": dict|None}
        หรือ ({"error": "No data"}, 400) ถ้าไม่มี body
    """
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