# RescuOpt AI

ระบบการคาดเดาความรุนแรงของน้ำท่วม และคำนวณเส้นทางการอพยพที่ปลอดภัยที่สุด โดยการ optimization

จัดทำโดย : นนอ.อริญชย์ หุนตระนี และ นนอ.อภิรักษ์ สาจันทร์

## Quick Start

### One-click launch (Windows)

```bat
start_rescuopt.bat
```

### Manual launch

**Step 1: Start the server**

```bash
conda activate geoai
python Server.py
```

Open browser: <http://localhost:5000/disaster_nav>

**Step 2: Start flood detection (new terminal)**

```bash
conda activate geoai
cd Flood-detection
python main.py
```

## Architecture

| Component | Role |
|-----------|------|
| `Server.py` | Flask backend — relay API for YOLO detections |
| `dashboard/disaster_nav.html` | Frontend UI — Leaflet map + 8 pathfinding algorithms (client-side) |
| `Flood-detection/main.py` | Tkinter YOLO GUI — camera/video flood detection |
| `Flood-detection/yolo.py` | YOLO video processor module |
| `Flood-detection/best.pt` | YOLOv8 model weights |

## Algorithms

**Search:** A*, BFS, Greedy Best-First
**Optimization:** Simulated Annealing, Hill Climbing (Steepest), Genetic Algorithm
**Constraint:** Backtracking, AC-3 Arc Consistency
