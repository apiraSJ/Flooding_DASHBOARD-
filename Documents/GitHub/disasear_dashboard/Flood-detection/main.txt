"""
Flood Detection Tkinter GUI Application
========================================
Interactive desktop application for real-time flood detection using YOLOv8.
Processes images and videos to detect flooded areas and provide severity assessments.

GUI Features:
    - Image/video file upload with preview
    - Real-time flood detection with confidence scores
    - Severity classification (Low/Medium/High)
    - Actionable survival recommendations based on severity
    - Beautiful dark-themed UI with modern design

Key Components:
    - FloodApp: Main application window
    - RoundedFrame: Custom rounded rectangle canvas widget
    - SeverityBar: Custom progress bar for severity visualization
    - YOLO Integration: Real-time inference on CPU/GPU
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading
import os
import requests
import webbrowser
import json
import functools
import math
import tkintermapview
from math import radians, sin, cos, sqrt, atan2
# ============================================================
#  Color and Font Constants
# ============================================================
BG          = "#0F1923"   # Main background (very dark blue)
SURFACE     = "#1A2634"   # Card surface
BORDER      = "#2A3A4A"   # Border color
ACCENT      = "#2196F3"   # Primary blue
TEXT_PRI    = "#E8F0F8"   # Primary text
TEXT_SEC    = "#7A9BB5"   # Secondary text

SEV_LOW     = "#4CAF50"   # Green — low severity
SEV_MID     = "#FF9800"   # Orange — medium severity
SEV_HIGH    = "#F44336"   # Red — high severity

FONT_TITLE  = ("Segoe UI", 20, "bold")
FONT_HEAD   = ("Segoe UI", 13, "bold")
FONT_BODY   = ("Segoe UI", 11)
FONT_SMALL  = ("Segoe UI", 10)
FONT_BADGE  = ("Segoe UI", 9, "bold")
EARTH_RADIUS = 6371000  # meters
DEFAULT_HAZARD_RADIUS_M = 500
SURVIVOR_COLLISION_DISTANCE_M = 5
CONF_THRESHOLD = 0.5
MIN_ANALYSIS_RESOLUTION = 640
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
# ============================================================
#  YOLO Integration Logic
# ============================================================
def run_yolo(file_path, file_type, on_result):
    """
    Run YOLOv8 model on image or video file and return flood detection results.
    
    Args:
        file_path (str): Path to image or video file to process
        file_type (str): Type of file - 'image' or 'video'
        on_result (callable): Callback function with signature:
            on_result(level, pct, title, advice) where:
            - level (str): "low" | "mid" | "high" | "error"
            - pct (int): 0-100 confidence/severity percentage
            - title (str): Severity level description
            - advice (str): Actionable survival recommendations
    
    Processing:
        1. Load YOLOv8 segmentation model (best.pt)
        2. For images: Direct inference
        3. For videos: Extract and analyze first frame
        4. Calculate maximum confidence from detected objects
        5. Classify severity based on confidence threshold
        6. Generate context-specific advice for users
    
    Severity Levels:
        - Low (<40% confidence): Normal operations, monitor updates
        - Medium (40-70%): Evacuate valuables, avoid water crossings
        - High (>70%): Immediate evacuation recommended
    
    Error Handling:
        - Invalid file format → returns error callback
        - Missing model file → raises exception
        - Video read failure → returns error callback
    """
    try:
        import torch
        import torch.serialization
        from ultralytics import YOLO
        from ultralytics.nn.tasks import SegmentationModel

        torch.serialization.add_safe_globals([SegmentationModel])
        model = YOLO("D:/geoai_train/geoai_train/Flood-detection/best.pt")
        model.to("cpu")

        if file_type == "image":
            results = model(file_path)
        else:
            # วิดีโอ — ใช้เฟรมแรก
            import cv2
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError("อ่านวิดีโอไม่ได้")
            results = model(frame)

        # คำนวณเปอร์เซ็นต์พื้นที่น้ำท่วมเทียบกับรูปทั้งหมด (Area Ratio)
        boxes_xyxy = results[0].boxes.xyxy.tolist() if results[0].boxes else []
        if boxes_xyxy:
            from shapely.geometry import box
            from shapely.ops import unary_union
            
            # ขนาดรูปภาพต้นฉบับ
            h, w = results[0].orig_shape
            total_area = float(w * h)
            
            # วาดกรอบ Polygon แล้วนำมา Union กันเพื่อหาพื้นที่รวมแบบไม่ซ้อนทับ
            polygons = [box(b[0], b[1], b[2], b[3]) for b in boxes_xyxy]
            union_poly = unary_union(polygons)
            flood_area = union_poly.area
            
            pct = int((flood_area / total_area) * 100)
            pct = min(100, max(0, pct))
        else:
            pct = 0

        if pct < 50:
            level, title = "low",  "น้ำท่วมระดับเบา"
            advice = ("ระดับน้ำอยู่ในเกณฑ์ต่ำ ยังสามารถสัญจรได้ตามปกติ\n"
                    "แนะนำติดตามข่าวสารและเตรียมพร้อมหากระดับน้ำเพิ่มขึ้น")
        elif pct <= 80:
            level, title = "mid",  "น้ำท่วมระดับปานกลาง"
            advice = ("ระดับน้ำมีผลกระทบต่อการสัญจรและที่พักอาศัยบางส่วน\n"
                    "ควรย้ายทรัพย์สินขึ้นที่สูง และหลีกเลี่ยงการขับรถลุยน้ำ")
        else:
            level, title = "high", "น้ำท่วมระดับรุนแรง"
            advice = ("อยู่ในสถานการณ์อันตราย ควรอพยพออกจากพื้นที่ทันที\n"
                    "ติดต่อหน่วยกู้ภัย โทร 1784 หรือ 199")

        on_result(level, pct, title, advice)

    except Exception as e:
        on_result("error", 0, "เกิดข้อผิดพลาด", str(e))


def run_yolo(file_path, file_type, on_result):
    """Validate input, run YOLO, and stop safely when analysis has no flood result."""
    try:
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError("Input file not found.")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        import cv2
        import torch
        import torch.serialization
        from shapely.geometry import box
        from shapely.ops import unary_union
        from ultralytics import YOLO
        from ultralytics.nn.tasks import SegmentationModel

        torch.serialization.add_safe_globals([SegmentationModel])
        model = YOLO(MODEL_PATH)
        model.to("cpu")

        if file_type == "image":
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Unable to read image. The file may be corrupted or unsupported.")
            height, width = image.shape[:2]
            if width < MIN_ANALYSIS_RESOLUTION or height < MIN_ANALYSIS_RESOLUTION:
                raise ValueError("Image resolution is too low. Minimum supported size is 640 x 640 pixels.")
            results = model(image, conf=CONF_THRESHOLD)
        else:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                cap.release()
                raise ValueError("Unable to open video.")
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise ValueError("Unable to read the first video frame.")
            height, width = frame.shape[:2]
            if width < MIN_ANALYSIS_RESOLUTION or height < MIN_ANALYSIS_RESOLUTION:
                raise ValueError("Video resolution is too low. Minimum supported size is 640 x 640 pixels.")
            results = model(frame, conf=CONF_THRESHOLD)

        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or len(boxes) == 0 or masks is None:
            on_result(
                "none",
                0,
                "No flood detected",
                "No flooded area was detected above the confidence threshold. Analysis stopped safely."
            )
            return

        raw_boxes = boxes.xyxy.tolist() if getattr(boxes, "xyxy", None) is not None else []
        raw_confs = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else [1.0] * len(raw_boxes)
        boxes_xyxy = [
            detection_box
            for detection_box, confidence in zip(raw_boxes, raw_confs)
            if float(confidence) >= CONF_THRESHOLD
        ]

        if not boxes_xyxy:
            on_result(
                "none",
                0,
                "No flood detected",
                "Detections were below the confidence threshold. Analysis stopped safely."
            )
            return

        orig_h, orig_w = result.orig_shape
        total_area = float(orig_w * orig_h)
        if total_area <= 0:
            raise ValueError("Invalid image dimensions returned by model.")

        polygons = []
        for b in boxes_xyxy:
            if len(b) < 4:
                continue
            candidate = box(b[0], b[1], b[2], b[3])
            if not candidate.is_empty and candidate.area > 0:
                polygons.append(candidate)

        if not polygons:
            on_result(
                "none",
                0,
                "No flood detected",
                "No valid flood geometry was produced by the model. Analysis stopped safely."
            )
            return

        flood_area = unary_union(polygons).area
        pct = min(100, max(0, int((flood_area / total_area) * 100)))

        if pct < 50:
            level, title = "low", "Flood severity: Low"
            advice = "Flood impact appears limited. Continue monitoring and stay prepared."
        elif pct <= 80:
            level, title = "mid", "Flood severity: Medium"
            advice = "Flooding may affect travel and shelter access. Avoid flooded roads and move valuables higher."
        else:
            level, title = "high", "Flood severity: High"
            advice = "Dangerous flooding detected. Evacuate immediately and contact rescue services if needed."

        on_result(level, pct, title, advice)

    except Exception as e:
        print(f"Analysis error: {e}")
        on_result("error", 0, "Analysis Error", str(e))


# ============================================================
#  Custom Widget Components
# ============================================================
class RoundedFrame(tk.Canvas):
    """Custom rounded rectangle canvas widget for modern UI.
    
    Creates visually appealing containers with rounded corners and borders.
    Used throughout the application for card-like UI elements.
    
    Attributes:
        _r (int): Corner radius in pixels
        _bg (str): Background color hex code
        _bd (str): Border color hex code
    """
    
    def __init__(self, parent, radius=12, bg=SURFACE, bd_color=BORDER, **kw):
        """Initialize rounded frame widget.
        
        Args:
            parent: Tkinter parent widget
            radius (int): Corner radius size (default 12 pixels)
            bg (str): Background color hex (default SURFACE color)
            bd_color (str): Border color hex (default BORDER color)
            **kw: Additional Canvas keyword arguments
        """
        super().__init__(parent, highlightthickness=0, bg=BG, **kw)
        self._r  = radius
        self._bg = bg
        self._bd = bd_color
        self.bind("<Configure>", self._redraw)

    def _redraw(self, _=None):
        """Redraw rounded rectangle on widget size change."""
        w, h, r = self.winfo_width(), self.winfo_height(), self._r
        self.delete("bg")
        self.create_rounded_rect(2, 2, w-2, h-2, r, fill=self._bg,
                                outline=self._bd, width=1, tags="bg")

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kw):
        """Create a smooth rounded rectangle polygon.
        
        Args:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            r: Corner radius
            **kw: Canvas create_polygon keyword arguments
        """
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
            x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
            x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self.create_polygon(pts, smooth=True, **kw)


class SeverityBar(tk.Canvas):
    """Custom progress bar for displaying flood severity level.
    
    Visual indicator showing flood severity from 0-100%.
    Color changes based on severity level:
    - Green (low): <40%
    - Orange (medium): 40-70%
    - Red (high): >70%
    
    Attributes:
        _pct (int): Current percentage (0-100)
        _color (str): Current bar color hex code
    """
    
    def __init__(self, parent, **kw):
        """Initialize severity bar widget.
        
        Args:
            parent: Tkinter parent widget
            **kw: Canvas keyword arguments
        """
        kw.pop("bg", None)
        super().__init__(parent, height=10, highlightthickness=0, bg=BG, **kw)
        self._pct   = 0
        self._color = SEV_LOW
        self.bind("<Configure>", self._draw)

    def set(self, pct, color):
        """Update bar percentage and color.
        
        Args:
            pct (int): Percentage value (0-100)
            color (str): Hex color code for bar fill
        """
        self._pct   = pct
        self._color = color
        self._draw()

    def _draw(self, _=None):
        """Redraw severity bar at current percentage and color."""
        w = self.winfo_width()
        h = self.winfo_height()
        self.delete("all")
        # track background
        self.create_rounded_rect(0, 0, w, h, 5, fill=BORDER, outline="")
        # filled portion
        fill_w = int(w * self._pct / 100)
        if fill_w > 0:
            self.create_rounded_rect(0, 0, fill_w, h, 5,
                                    fill=self._color, outline="")

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kw):
        """Create smooth rounded rectangle for progress bar.
        
        Args:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            r: Corner radius
            **kw: Canvas create_polygon keyword arguments
        """
        r = min(r, (x2-x1)//2, (y2-y1)//2)
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
            x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
            x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self.create_polygon(pts, smooth=True, **kw)


# ============================================================
#  Main Application Window
# ============================================================
class FloodApp(tk.Tk):
    """Main application window for flood detection GUI.
    
    Manages user interface for:
    - File selection (image/video)
    - Real-time YOLO inference
    - Result display with severity indicators
    - Actionable recommendations for users
    
    Architecture:
        - Main window with dark theme
        - Upload panel for file selection
        - Preview area for images
        - Results panel with severity gauge
        - Advice panel with survival tips
    """
    
    def __init__(self):
        """Initialize main application window."""
        super().__init__()
        self.title("ระบบตรวจจับน้ำท่วม")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.file_path  = None
        self.file_type  = tk.StringVar(value="image")
        self._thumb_ref = None   # กัน GC

        # Map picker data (replaces manual lat/lng entry)
        self.map_survivors = []  # [{id, lat, lng}, ...]
        self.map_hazards = []    # [{type, lat, lng, severity, radius_m}, ...]
        self.analyzed_severity = 5
        self.analyzed_radius_m = DEFAULT_HAZARD_RADIUS_M

        self._build_ui()
        self._center_window(680, 800)

    # ── จัดหน้าต่างให้อยู่กลางจอ ──────────────────────────
    def _center_window(self, w, h):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    # ── สร้าง UI ───────────────────────────────────────────
    def _build_ui(self):
        root = tk.Frame(self, bg=BG, padx=28, pady=24)
        root.pack(fill="both", expand=True)

        # ── Header ──────────────────────────────────────────
        hdr = tk.Frame(root, bg=BG)
        hdr.pack(fill="x", pady=(0, 18))

        badge = tk.Label(hdr, text="  💧 ระบบตรวจจับน้ำท่วม  ",
                         font=FONT_BADGE, bg="#1A3A5C", fg=ACCENT,
                         padx=8, pady=4)
        badge.pack(anchor="w")
        tk.Label(hdr, text="วิเคราะห์ความรุนแรงน้ำท่วม",
                 font=FONT_TITLE, bg=BG, fg=TEXT_PRI).pack(anchor="w", pady=(6,2))
        tk.Label(hdr, text="อัปโหลดภาพหรือวิดีโอเพื่อประเมินระดับความรุนแรง",
                 font=FONT_SMALL, bg=BG, fg=TEXT_SEC).pack(anchor="w")

        # ── Toggle ภาพ / วิดีโอ ─────────────────────────────
        tog = tk.Frame(root, bg=BG)
        tog.pack(fill="x", pady=(0, 14))

        self.btn_img = self._toggle_btn(tog, "📷  ภาพนิ่ง",  "image")
        self.btn_vid = self._toggle_btn(tog, "🎬  วิดีโอ",   "video")
        self.btn_img.pack(side="left", padx=(0, 8))
        self.btn_vid.pack(side="left")
        self._highlight_toggle()

        # ── เลือกตำแหน่งจากแผนที่ ──────────────────────────────
        map_card = tk.Frame(root, bg=SURFACE,
                            highlightbackground=BORDER, highlightthickness=1)
        map_card.pack(fill="x", pady=(0, 14))

        map_inner = tk.Frame(map_card, bg=SURFACE, padx=16, pady=14)
        map_inner.pack(fill="both")

        map_title_row = tk.Frame(map_inner, bg=SURFACE)
        map_title_row.pack(fill="x", pady=(0, 10))
        tk.Label(map_title_row, text="🗺️", font=("Segoe UI", 18),
                 bg=SURFACE).pack(side="left")
        tk.Label(map_title_row, text="ตำแหน่งจากแผนที่",
                 font=FONT_HEAD, bg=SURFACE, fg=TEXT_PRI).pack(side="left", padx=(8, 0))

        # Summary counts
        self.map_summary_frame = tk.Frame(map_inner, bg=SURFACE)
        self.map_summary_frame.pack(fill="x", pady=(0, 10))

        surv_box = tk.Frame(self.map_summary_frame, bg="#1A3A2A", padx=10, pady=6)
        surv_box.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.surv_count_lbl = tk.Label(surv_box, text="👤 ผู้รอดชีวิต: 0 จุด",
                                        font=FONT_SMALL, bg="#1A3A2A", fg=SEV_LOW)
        self.surv_count_lbl.pack()

        haz_box = tk.Frame(self.map_summary_frame, bg="#3A1A1A", padx=10, pady=6)
        haz_box.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.haz_count_lbl = tk.Label(haz_box, text="⚠️ ภัยพิบัติ: 0 จุด",
                                       font=FONT_SMALL, bg="#3A1A1A", fg=SEV_HIGH)
        self.haz_count_lbl.pack()

        # Open map button
        self.map_btn = tk.Button(map_inner,
                                  text="🗺️  เปิดแผนที่เลือกตำแหน่ง",
                                  font=FONT_HEAD, fg="#FFFFFF", bg="#1565C0",
                                  activeforeground="#FFFFFF", activebackground="#0D47A1",
                                  relief="flat", cursor="hand2", bd=0,
                                  padx=0, pady=10,
                                  command=self._open_map_picker)
        self.map_btn.pack(fill="x")

        self.map_status_lbl = tk.Label(map_inner, text="",
                                        font=FONT_SMALL, bg=SURFACE, fg=TEXT_SEC)
        self.map_status_lbl.pack(pady=(6, 0))

        # ── Drop Zone ───────────────────────────────────────
        self.drop_frame = tk.Frame(root, bg=BORDER, pady=1)
        self.drop_frame.pack(fill="x", pady=(0, 14))

        self.drop_inner = tk.Frame(self.drop_frame, bg=SURFACE, padx=20, pady=30)
        self.drop_inner.pack(fill="both", padx=1, pady=1)
        self.drop_inner.bind("<Button-1>", lambda _: self._pick_file())

        self.drop_icon = tk.Label(self.drop_inner, text="☁", font=("Segoe UI", 40),
                                  bg=SURFACE, fg=TEXT_SEC, cursor="hand2")
        self.drop_icon.pack()
        self.drop_icon.bind("<Button-1>", lambda _: self._pick_file())

        self.drop_text = tk.Label(self.drop_inner,
                                  text="คลิกเพื่อเลือกไฟล์",
                                  font=FONT_HEAD, bg=SURFACE, fg=TEXT_PRI, cursor="hand2")
        self.drop_text.pack(pady=(4, 2))
        self.drop_text.bind("<Button-1>", lambda _: self._pick_file())

        self.drop_sub = tk.Label(self.drop_inner, text="รองรับ JPG, PNG, WEBP",
                                 font=FONT_SMALL, bg=SURFACE, fg=TEXT_SEC)
        self.drop_sub.pack()

        # ── Preview ─────────────────────────────────────────
        self.preview_frame = tk.Frame(root, bg=BG)
        # (ซ่อนไว้ก่อน)

        self.preview_label = tk.Label(self.preview_frame, bg=SURFACE,
                                      relief="flat", cursor="hand2")
        self.preview_label.pack()
        self.preview_label.bind("<Button-1>", lambda _: self._pick_file())

        self.remove_btn = tk.Button(self.preview_frame,
                                    text="✕  เปลี่ยนไฟล์",
                                    font=FONT_SMALL, fg=TEXT_SEC, bg=SURFACE,
                                    activeforeground=TEXT_PRI, activebackground=BORDER,
                                    relief="flat", cursor="hand2", bd=0,
                                    command=self._remove_file)
        self.remove_btn.pack(pady=(6, 0))

        # ── ปุ่มวิเคราะห์ ────────────────────────────────────
        self.analyze_btn = tk.Button(root, text="🔍  วิเคราะห์ภาพ",
                                     font=FONT_HEAD, fg="#FFFFFF", bg=ACCENT,
                                     activeforeground="#FFFFFF", activebackground="#1565C0",
                                     relief="flat", cursor="hand2", bd=0,
                                     padx=0, pady=12,
                                     state="disabled",
                                     command=self._start_analysis)
        self.analyze_btn.pack(fill="x", pady=(0, 14))

        # ── Loading dots ─────────────────────────────────────
        self.loading_frame = tk.Frame(root, bg=BG)
        self.dots_label = tk.Label(self.loading_frame, text="",
                                   font=("Segoe UI", 22), bg=BG, fg=ACCENT)
        self.dots_label.pack()
        self._dot_state = 0

        # ── Result card ──────────────────────────────────────
        self.result_frame = tk.Frame(root, bg=BG)

        res_card = tk.Frame(self.result_frame, bg=SURFACE,
                            highlightbackground=BORDER, highlightthickness=1)
        res_card.pack(fill="x")

        inner = tk.Frame(res_card, bg=SURFACE, padx=18, pady=16)
        inner.pack(fill="both")

        # icon + title row
        top_row = tk.Frame(inner, bg=SURFACE)
        top_row.pack(fill="x", pady=(0, 14))

        self.res_icon  = tk.Label(top_row, text="", font=("Segoe UI", 26),
                                  bg=SURFACE, width=2)
        self.res_icon.pack(side="left")

        meta = tk.Frame(top_row, bg=SURFACE)
        meta.pack(side="left", padx=(10, 0))
        tk.Label(meta, text="ผลการวิเคราะห์", font=FONT_SMALL,
                 bg=SURFACE, fg=TEXT_SEC).pack(anchor="w")
        self.res_title = tk.Label(meta, text="", font=FONT_HEAD,
                                  bg=SURFACE, fg=TEXT_PRI)
        self.res_title.pack(anchor="w")

        # severity bar
        bar_row = tk.Frame(inner, bg=SURFACE)
        bar_row.pack(fill="x", pady=(0, 8))
        bar_lbl = tk.Frame(bar_row, bg=SURFACE)
        bar_lbl.pack(fill="x", pady=(0, 5))
        tk.Label(bar_lbl, text="ระดับความรุนแรง", font=FONT_SMALL,
                 bg=SURFACE, fg=TEXT_SEC).pack(side="left")
        self.sev_pct_lbl = tk.Label(bar_lbl, text="", font=FONT_SMALL,
                                    bg=SURFACE, fg=TEXT_SEC)
        self.sev_pct_lbl.pack(side="right")
        self.sev_bar = SeverityBar(inner, bg=BG)
        self.sev_bar.pack(fill="x", pady=(0, 12))

        # chips
        chips_row = tk.Frame(inner, bg=SURFACE)
        chips_row.pack(fill="x", pady=(0, 12))
        self.chip_low  = self._chip(chips_row, "เบา")
        self.chip_mid  = self._chip(chips_row, "ปานกลาง")
        self.chip_high = self._chip(chips_row, "รุนแรง")
        for c in (self.chip_low, self.chip_mid, self.chip_high):
            c.pack(side="left", expand=True, fill="x", padx=4)

        # advice
        advice_bg = tk.Frame(inner, bg="#0F1923",
                             highlightbackground=BORDER, highlightthickness=1)
        advice_bg.pack(fill="x")
        self.advice_lbl = tk.Label(advice_bg, text="", font=FONT_BODY,
                                   bg="#0F1923", fg=TEXT_SEC,
                                   justify="left", wraplength=560,
                                   padx=14, pady=12)
        self.advice_lbl.pack(anchor="w")

        # save button
        self.save_btn = tk.Button(self.result_frame,
                                  text="💾  บันทึกวิดีโอผลลัพธ์",
                                  font=FONT_BODY, fg=TEXT_PRI, bg=SURFACE,
                                  activeforeground=TEXT_PRI, activebackground=BORDER,
                                  relief="flat", cursor="hand2", bd=0,
                                  pady=10, highlightthickness=1,
                                  highlightbackground=BORDER,
                                  command=self._open_output)
        self.save_btn.pack(fill="x", pady=(10, 0))
        self.save_btn.pack_forget()

    # ── helpers ────────────────────────────────────────────
    def _toggle_btn(self, parent, text, value):
        btn = tk.Button(parent, text=text, font=FONT_SMALL,
                        relief="flat", bd=0, padx=14, pady=7,
                        cursor="hand2",
                        command=lambda v=value: self._set_type(v))
        return btn

    def _highlight_toggle(self):
        t = self.file_type.get()
        for btn, val in ((self.btn_img, "image"), (self.btn_vid, "video")):
            if val == t:
                btn.configure(bg=ACCENT, fg="#FFFFFF",
                              activebackground="#1565C0", activeforeground="#FFFFFF")
            else:
                btn.configure(bg=SURFACE, fg=TEXT_SEC,
                              activebackground=BORDER, activeforeground=TEXT_PRI)

    def _chip(self, parent, text):
        return tk.Label(parent, text=text, font=FONT_BADGE,
                        bg=BORDER, fg=TEXT_SEC, padx=0, pady=5, relief="flat")

    def _set_type(self, t):
        self.file_type.set(t)
        self._highlight_toggle()
        sub = "รองรับ JPG, PNG, WEBP" if t == "image" else "รองรับ MP4, AVI"
        self.drop_sub.configure(text=sub)
        self._remove_file()

    # ── ไฟล์ ───────────────────────────────────────────────
    def _pick_file(self):
        if self.file_type.get() == "image":
            ft = [("ไฟล์ภาพ", "*.jpg *.jpeg *.png *.webp *.bmp")]
        else:
            ft = [("ไฟล์วิดีโอ", "*.mp4 *.avi *.mov *.mkv")]
        path = filedialog.askopenfilename(filetypes=ft)
        if path:
            self.file_path = path
            self._show_preview(path)

    def _show_preview(self, path):
        self.drop_frame.pack_forget()
        self.result_frame.pack_forget()
        self.loading_frame.pack_forget()

        if self.file_type.get() == "image":
            try:
                img = Image.open(path)
                img.thumbnail((624, 260), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._thumb_ref = photo
                self.preview_label.configure(image=photo, text="")
            except Exception:
                self.preview_label.configure(image="",
                                             text="[ไม่สามารถแสดงตัวอย่างได้]",
                                             font=FONT_BODY, fg=TEXT_SEC)
        else:
            name = os.path.basename(path)
            self.preview_label.configure(image="",
                                         text=f"🎬  {name}",
                                         font=FONT_HEAD, fg=TEXT_PRI,
                                         padx=20, pady=40)

        self.preview_frame.pack(fill="x", pady=(0, 14))
        self.analyze_btn.configure(state="normal", text="🔍  วิเคราะห์ภาพ")

    def _remove_file(self):
        self.file_path = None
        self.preview_frame.pack_forget()
        self.result_frame.pack_forget()
        self.loading_frame.pack_forget()
        self.drop_frame.pack(fill="x", pady=(0, 14))
        self.analyze_btn.configure(state="disabled")
        self.preview_label.configure(image="")
        self._thumb_ref = None

    # ── วิเคราะห์ ───────────────────────────────────────────
    def _start_analysis(self):
        if not self.file_path:
            return
        self.analyze_btn.configure(state="disabled", text="กำลังวิเคราะห์...")
        self.result_frame.pack_forget()
        self.loading_frame.pack(fill="x", pady=20)
        self._animate_dots()

        t = threading.Thread(target=run_yolo,
                             args=(self.file_path, self.file_type.get(),
                                   self._on_result), daemon=True)
        t.start()

    def _animate_dots(self):
        frames = ["●○○", "○●○", "○○●", "○●○"]
        if self.loading_frame.winfo_ismapped():
            self.dots_label.configure(text=frames[self._dot_state % 4])
            self._dot_state += 1
            self.after(300, self._animate_dots)

    def _on_result(self, level, pct, title, advice):
        self.after(0, lambda: self._show_result(level, pct, title, advice))

    def _show_result(self, level, pct, title, advice):
        self.loading_frame.pack_forget()
        self.analyze_btn.configure(state="normal", text="🔍  วิเคราะห์ใหม่")

        if level == "error":
            self.res_icon.configure(text="⚠️")
            self.res_title.configure(text=title, fg="#F44336")
            self.advice_lbl.configure(text=advice)
            self.sev_bar.set(0, BORDER)
            self.sev_pct_lbl.configure(text="")
            for c in (self.chip_low, self.chip_mid, self.chip_high):
                c.configure(bg=BORDER, fg=TEXT_SEC)
            self.save_btn.pack_forget()
        else:
            icons   = {"low": "💧", "mid": "🌊", "high": "🚨"}
            colors  = {"low": SEV_LOW, "mid": SEV_MID, "high": SEV_HIGH}
            color   = colors[level]

            self.res_icon.configure(text=icons[level])
            self.res_title.configure(text=title, fg=color)
            self.sev_pct_lbl.configure(text=f"{pct}%")
            self.sev_bar.set(pct, color)
            self.advice_lbl.configure(text=advice)

            # chips
            chip_map = {"low": self.chip_low,
                        "mid": self.chip_mid,
                        "high": self.chip_high}
            for k, chip in chip_map.items():
                if k == level:
                    chip.configure(bg=color, fg="#FFFFFF")
                else:
                    chip.configure(bg=BORDER, fg=TEXT_SEC)

            # แสดงปุ่มบันทึกเฉพาะวิดีโอ
            if self.file_type.get() == "video" and os.path.exists("Output Video.mp4"):
                self.save_btn.pack(fill="x", pady=(10, 0))
            else:
                self.save_btn.pack_forget()

            # อัปเดตขนาดวงน้ำท่วมอัตโนมัติตามความรุนแรง
            sev_map = {"low": 3, "mid": 6, "high": 9}
            rad_map = {"low": 2000, "mid": 20000, "high": 100000}
            self.analyzed_severity = sev_map.get(level, 5)
            self.analyzed_radius_m = rad_map.get(level, 500)

            # นำค่าที่คำนวณได้ไปอัปเดตจุดภัยพิบัติที่ผู้ใช้ปักไว้
            for h in self.map_hazards:
                h["severity"] = self.analyzed_severity
                h["radius_m"] = self.analyzed_radius_m

            # ส่งข้อมูลไป Server (ใช้ข้อมูลจาก Map Picker)
            try:
                payload = {
                    "survivors": self.map_survivors,
                    "hazards": self.map_hazards,
                    "severity": self.analyzed_severity,
                    "confidence": pct,
                    "level_label": title
                }
                res = requests.post("http://localhost:5000/api/report_flood_v2", json=payload)
                if res.status_code == 200:
                    messagebox.showinfo("สำเร็จ", "ส่งข้อมูลไปยัง Dashboard สำเร็จ")
                    webbrowser.open("http://localhost:5000")
                else:
                    messagebox.showwarning("แจ้งเตือน", f"ส่งข้อมูลไม่สำเร็จ: {res.status_code}")
            except Exception as e:
                messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถเชื่อมต่อ Server ได้: {e}")

        self.result_frame.pack(fill="x")

    def _open_output(self):
        path = os.path.abspath("Output Video.mp4")
        if os.path.exists(path):
            os.startfile(path)

    # ── Embedded Map Picker (tkintermapview) ─────────────────
    def _open_map_picker(self):
        """Open embedded map picker dialog."""
        MapPickerDialog(self, self.map_survivors, self.map_hazards, self._on_map_data, self.analyzed_severity, self.analyzed_radius_m)

    def _on_map_data(self, survivors, hazards):
        """Callback when map picker sends confirmed data."""
        self.map_survivors = survivors
        self.map_hazards = hazards

        s_count = len(survivors)
        h_count = len(hazards)

        self.surv_count_lbl.configure(text=f"👤 ผู้รอดชีวิต: {s_count} จุด")
        self.haz_count_lbl.configure(text=f"⚠️ ภัยพิบัติ: {h_count} จุด")

        status_parts = []
        if s_count > 0:
            status_parts.append(f"{s_count} ผู้รอดชีวิต")
        if h_count > 0:
            status_parts.append(f"{h_count} ภัยพิบัติ")

        self.map_status_lbl.configure(
            text=f"✅ ได้รับข้อมูล: {', '.join(status_parts)}" if status_parts else "ไม่มีข้อมูล",
            fg=SEV_LOW if status_parts else TEXT_SEC
        )
        self.map_btn.configure(text="🗺️  แก้ไขตำแหน่งบนแผนที่")

class MapPickerDialog(tk.Toplevel):
    def __init__(self, parent, initial_survivors, initial_hazards, callback, current_severity, current_radius_m):
        super().__init__(parent)
        self.title("🗺️ Map Picker")
        self.geometry("900x700")
        self.configure(bg=BG)
        self.transient(parent)
        self.grab_set()

        self.survivors = list(initial_survivors)
        self.hazards = list(initial_hazards)
        self.callback = callback
        self.current_severity = current_severity
        self.current_radius_m = current_radius_m
        
        self.mode = None  # "survivor" or "hazard"
        self._map_markers = []
        self._map_paths = []

        self._build_ui()
        self._render_all()

    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self, bg=SURFACE, padx=10, pady=10)
        toolbar.pack(fill="x")

        # Survivor Mode btn
        self.btn_surv = tk.Button(toolbar, text="👤 เพิ่มผู้รอดชีวิต", font=FONT_HEAD,
                                   bg=SURFACE, fg=TEXT_PRI, command=lambda: self.set_mode("survivor"))
        self.btn_surv.pack(side="left", padx=5)

        # Hazard Mode btn
        self.btn_haz = tk.Button(toolbar, text="⚠️ เพิ่มภัยพิบัติ", font=FONT_HEAD,
                                  bg=SURFACE, fg=TEXT_PRI, command=lambda: self.set_mode("hazard"))
        self.btn_haz.pack(side="left", padx=5)

        # Hazard Config frame (Only type left)
        self.haz_cfg_frame = tk.Frame(toolbar, bg=SURFACE)
        tk.Label(self.haz_cfg_frame, text="ชนิด:", bg=SURFACE, fg=TEXT_PRI).pack(side="left", padx=2)
        self.haz_type_cb = ttk.Combobox(self.haz_cfg_frame, values=["flood", "fire", "collapse", "toxic", "earthquake"], width=10, state="readonly")
        self.haz_type_cb.set("flood")
        self.haz_type_cb.pack(side="left", padx=2)

        self.lbl_status = tk.Label(toolbar, text="เลือกโหมดเพื่อคลิกวางจุดบนแผนที่", bg=SURFACE, fg=TEXT_SEC, font=FONT_BODY)
        self.lbl_status.pack(side="left", padx=20)

        btn_clear = tk.Button(toolbar, text="🗑️ ล้างทั้งหมด", font=FONT_BODY, bg=BORDER, fg=TEXT_PRI, command=self._clear_all)
        btn_clear.pack(side="right", padx=5)

        btn_confirm = tk.Button(toolbar, text="✅ ยืนยัน", font=FONT_HEAD, bg=ACCENT, fg="#ffffff", command=self._confirm)
        btn_confirm.pack(side="right", padx=5)

        # Map Widget
        self.map_widget = tkintermapview.TkinterMapView(self, corner_radius=0)
        self.map_widget.pack(fill="both", expand=True)
        # Default center
        if self.survivors:
            self.map_widget.set_position(self.survivors[0]["lat"], self.survivors[0]["lng"])
        elif self.hazards:
            self.map_widget.set_position(self.hazards[0]["lat"], self.hazards[0]["lng"])
        else:
            self.map_widget.set_position(13.7563, 100.5018)  # BKK
        
        self.map_widget.set_zoom(13)
        self.map_widget.add_left_click_map_command(self._on_map_click)

    def set_mode(self, mode):
        self.mode = mode
        if mode == "survivor":
            self.btn_surv.configure(bg=SEV_LOW, fg="#ffffff")
            self.btn_haz.configure(bg=SURFACE, fg=TEXT_PRI)
            self.haz_cfg_frame.pack_forget()
            self.lbl_status.configure(text="👤 คลิกบนแผนที่เพื่อวางผู้รอดชีวิต", fg=SEV_LOW)
        elif mode == "hazard":
            self.btn_haz.configure(bg=SEV_HIGH, fg="#ffffff")
            self.btn_surv.configure(bg=SURFACE, fg=TEXT_PRI)
            self.haz_cfg_frame.pack(side="left", padx=10)
            self.lbl_status.configure(text="⚠️ กำหนดค่าแล้วคลิกเพื่อวางภัยพิบัติ", fg=SEV_HIGH)
    
    def haversine(self, lat1, lon1, lat2, lon2):
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = (
            sin(dlat / 2) ** 2
            + cos(radians(lat1)) * cos(radians(lat2))
            * sin(dlon / 2) ** 2
        )
        return EARTH_RADIUS * 2 * atan2(sqrt(a), sqrt(1 - a))

    def _current_hazard_radius(self):
        return self.current_radius_m or DEFAULT_HAZARD_RADIUS_M

    def _hazard_radius(self, hazard):
        return hazard.get("radius_m") or DEFAULT_HAZARD_RADIUS_M

    def survivor_inside_hazard(self, lat, lng):
        for hazard in self.hazards:

            d = self.haversine(
                lat,
                lng,
                hazard["lat"],
                hazard["lng"]
            )
            if d <= self._hazard_radius(hazard):
                return True

        return False
    def survivor_collision(self, lat, lng):
        for survivor in self.survivors:
            d = self.haversine(
                lat,
                lng,
                survivor["lat"],
                survivor["lng"]
            )
            if d <= SURVIVOR_COLLISION_DISTANCE_M:
                return True

        return False

    def hazard_overlap(self, lat, lng):

        for hazard in self.hazards:

            d = self.haversine(
                lat,
                lng,
                hazard["lat"],
                hazard["lng"]
            )

            min_dist = max(
                self._hazard_radius(hazard),
                self._current_hazard_radius()
            )

            if d < min_dist:
                return True

        return False
    def hazard_on_survivor(self, lat, lng):

        for survivor in self.survivors:

            d = self.haversine(
                lat,
                lng,
                survivor["lat"],
                survivor["lng"]
            )

            if d < self._current_hazard_radius():
                return True

        return False
    
    def _on_map_click(self, coords):
        if not self.mode:
            return
        
        lat, lng = coords
        if self.mode == "survivor":
            if self.survivor_inside_hazard(lat, lng):
                messagebox.showwarning(
                    "Invalid Position",
                    "Survivor cannot be placed inside a hazard zone."
                )
                return
            if self.survivor_collision(lat, lng):
                messagebox.showwarning(
                    "Invalid Position",
                    "Survivor already exists at this location."
                )
                return
            self.survivors.append({
                "id": f"S{len(self.survivors)+1}",
                "lat": lat,
                "lng": lng
            })
        elif self.mode == "hazard":
            if self.hazard_overlap(lat, lng):
                messagebox.showwarning(
                    "Invalid Hazard",
                    "Hazard overlaps another hazard."
                )
                return

            if self.hazard_on_survivor(lat, lng):
                messagebox.showwarning(
                    "Invalid Hazard",
                    "Hazard cannot be placed over a survivor."
                )
                return

            self.hazards.append({
                "type": self.haz_type_cb.get(),
                "lat": lat,
                "lng": lng,
                "severity": self.current_severity,
                "radius_m": self._current_hazard_radius()
            })

        self._render_all()

    def _clear_all(self):
        self.survivors = []
        self.hazards = []
        self._render_all()

    def _render_all(self):
        self.map_widget.delete_all_marker()
        self.map_widget.delete_all_path()
        self.map_widget.delete_all_polygon()

        # Render survivors
        for s in self.survivors:
            self.map_widget.set_marker(s["lat"], s["lng"], text=f"👤 {s['id']}", text_color=SEV_LOW)

        # Render hazards
        for h in self.hazards:
            if h.get("radius_m"):
                # Analyzed -> draw full circle and details
                self.map_widget.set_marker(h["lat"], h["lng"], text=f"⚠️ {h['type'].upper()} (Sev:{h['severity']})", text_color=SEV_HIGH)
                points = self._get_circle_points(h["lat"], h["lng"], h["radius_m"])
                color_map = {"flood": "#3b82f6", "fire": "#f97316", "collapse": "#a855f7", "toxic": "#eab308", "earthquake": "#ec4899"}
                col = color_map.get(h["type"], "#ef4444")
                self.map_widget.set_polygon(points, fill_color=col, outline_color=col, border_width=2)
            else:
                # Not yet analyzed -> just a marker
                self.map_widget.set_marker(h["lat"], h["lng"], text=f"📍 {h['type'].upper()} (รอวิเคราะห์)", text_color=SEV_HIGH)

    def _get_circle_points(self, center_lat, center_lng, radius_m, num_points=36):
        # 1 deg lat = 111.32 km = 111320 m
        lat_offset = radius_m / 111320.0
        points = []
        for i in range(num_points):
            angle = math.radians(float(i) / num_points * 360.0)
            d_lat = math.sin(angle) * lat_offset
            d_lng = math.cos(angle) * lat_offset / math.cos(math.radians(center_lat))
            points.append((center_lat + d_lat, center_lng + d_lng))
        return points

    def _confirm(self):
        self.callback(self.survivors, self.hazards)
        self.destroy()


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    app = FloodApp()
    app.mainloop()
