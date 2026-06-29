"""
╔══════════════════════════════════════════════════════════════════╗
║        DISASTER SURVIVAL AI — OPTIMIZATION MODEL                 ║
║        อ้างอิง: lecture3_optimization.pdf                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Algorithms:                                                     ║
║    1. Hill Climbing  (Steepest / Random Restart)                 ║
║    2. Simulated Annealing                                        ║
║    3. A* Search                                                  ║
║    4. Greedy LP  (Resource Allocation)                           ║
║                                                                  ║
║  Objective Function:                                             ║
║    minimize  dist(path) x (1 + 4 x danger) + danger_penalty     ║
╚══════════════════════════════════════════════════════════════════╝

INPUT
-----
  hazards   : ประเภท, ความรุนแรง, พิกัด, รัศมี
  survivors : พิกัด, สุขภาพ, การเคลื่อนที่, เสบียง
  exit      : พิกัดทางออก
  algorithm : sa | hc | hc_restart | astar

OUTPUT
------
  เส้นทางที่ดีที่สุด + ต้นทุน + อันตราย + ลำดับผู้รอดชีวิต
  การจัดสรรทรัพยากร + คำแนะนำการเอาชีวิตรอด
"""

from __future__ import annotations
import math
import random
import heapq
import time
from dataclasses import dataclass, field


# ======================================================================
# 1.  CONSTANTS & SURVIVAL KNOWLEDGE BASE
# ======================================================================

HAZARD_TYPES = {"fire", "flood", "collapse", "toxic", "earthquake"}

HAZARD_LABELS = {
    "fire":       "FIRE - ไฟไหม้",
    "flood":      "FLOOD - น้ำท่วม",
    "collapse":   "COLLAPSE - อาคารถล่ม",
    "toxic":      "TOXIC - สารพิษ",
    "earthquake": "EARTHQUAKE - แผ่นดินไหว",
}

SURVIVAL_TIPS = {
    "fire": [
        "เคลื่อนที่ทวนลม (upwind) ออกจากแนวไฟเสมอ",
        "ลงต่ำใกล้พื้น ออกซิเจนอยู่ใต้ระดับควัน",
        "ปิดปากจมูกด้วยผ้าชุ่มน้ำป้องกันควันพิษ",
        "หลีกเลี่ยงอาคารที่อาจพังเพราะความร้อน",
        "อย่าเปิดประตูร้อน ตรวจสอบด้วยหลังมือก่อนเสมอ",
    ],
    "flood": [
        "อย่าเดินลุยน้ำที่ไหลเร็ว แม้ระดับตื้นก็พาล้มได้",
        "มุ่งสู่พื้นที่สูง ห่างจากร่องน้ำและถนนต่ำ",
        "ระวังสายไฟฟ้าที่ขาดตกลงในน้ำ",
        "หากติดในอาคาร ขึ้นชั้นบนสุดและส่งสัญญาณขอความช่วยเหลือ",
        "อย่าขับรถฝ่าน้ำท่วม น้ำ 60 ซม. พัดรถยนต์ได้",
    ],
    "collapse": [
        "DROP-COVER-HOLD: หมอบ หลบใต้โต๊ะแข็ง ยึดให้มั่น",
        "ห่างจากกระจก ของหนัก และชั้นวางของที่อาจล้มทับ",
        "อย่าใช้ลิฟต์ ใช้บันไดฉุกเฉินเท่านั้น",
        "หากติดใต้ซากปรักหักพัง เคาะท่อเป็นจังหวะส่งสัญญาณ",
        "ระวัง aftershock อย่าเข้าอาคารที่เสียหายก่อนตรวจสอบ",
    ],
    "toxic": [
        "ปิดปากจมูกด้วยผ้าหลายชั้นชุบน้ำเป็น mask ฉุกเฉิน",
        "เคลื่อนที่ตั้งฉากกับทิศทางลมออกจากกลุ่มควันพิษ",
        "อย่าสัมผัสสิ่งที่อาจปนเปื้อน ใช้เสื้อผ้าคลุมผิวหนัง",
        "ล้างร่างกายด้วยน้ำสะอาดจำนวนมากทันทีหากสัมผัสสาร",
        "ปิดประตูหน้าต่างและอุดช่องลมหากต้องอยู่ในอาคาร",
    ],
    "earthquake": [
        "อยู่ห่างจากอาคาร ต้นไม้ เสาไฟ และสายไฟ",
        "หากอยู่นอกอาคาร ออกสู่พื้นที่โล่งและหมอบราบกับพื้น",
        "ระวัง aftershock ที่อาจรุนแรงกว่าครั้งแรก",
        "ปิดวาล์วแก๊สทันทีหลังเกิดเหตุป้องกันไฟไหม้",
        "อย่าเข้าอาคารแตกร้าวจนกว่าผู้เชี่ยวชาญตรวจสอบ",
    ],
}


# ======================================================================
# 2.  DATA STRUCTURES
# ======================================================================

@dataclass
class Hazard:
    """
    ภัยพิบัติหนึ่งจุด
    ─────────────────
    danger_at(px, py) ใช้ linear decay:
      danger = (severity/10) x (1 - dist/radius)    ถ้า dist < radius
             = 0                                      ถ้า dist >= radius
    """
    hazard_type : str    # fire | flood | collapse | toxic | earthquake
    x           : float  # พิกัด X  (0-100)
    y           : float  # พิกัด Y  (0-100)
    severity    : int    # ความรุนแรง 1-10
    radius      : float  # รัศมีอันตราย

    def __post_init__(self):
        assert self.hazard_type in HAZARD_TYPES,  \
            f"hazard_type ต้องเป็นหนึ่งใน {HAZARD_TYPES}"
        assert 1 <= self.severity <= 10, "severity ต้องอยู่ระหว่าง 1-10"
        assert self.radius > 0,          "radius ต้องมากกว่า 0"

    def danger_at(self, px: float, py: float) -> float:
        """คืนค่า 0.0 (ปลอดภัย) ถึง 1.0 (อันตรายสูงสุด)"""
        dist = math.hypot(px - self.x, py - self.y)
        if dist >= self.radius:
            return 0.0
        return (self.severity / 10.0) * (1.0 - dist / self.radius)


@dataclass
class Survivor:
    """
    ผู้รอดชีวิต
    ────────────
    urgency_score = (100-health)x0.5 + (100-mobility)x0.3 + (100-supplies)x0.2
    ยิ่งสูงยิ่งต้องช่วยก่อน (ใช้ใน LP resource allocation)
    """
    survivor_id : str
    x           : float
    y           : float
    health      : int    # 0-100
    mobility    : int    # 0-100
    supplies    : int    # 0-100

    @property
    def urgency_score(self) -> float:
        return (
            (100 - self.health)   * 0.5 +
            (100 - self.mobility) * 0.3 +
            (100 - self.supplies) * 0.2
        )

    @property
    def status_label(self) -> str:
        u = self.urgency_score
        if u >= 60: return "CRITICAL - วิกฤต"
        if u >= 35: return "URGENT - ต้องการความช่วยเหลือ"
        return "STABLE - ปลอดภัยพอสมควร"


# ======================================================================
# 3.  DISASTER MAP
# ======================================================================

class DisasterMap:
    """
    พื้นที่ภัยพิบัติ — รวม danger field จากทุก hazard

    total_danger(x, y) = min(sum of all hazard.danger_at(x,y), 1.0)

    movement_cost(a, b) = dist(a,b) x (1 + 4 x danger_midpoint)
      ─ อันตรายทำให้เส้นทางแพงขึ้นสูงสุด 5 เท่า
      ─ AI จะอ้อมเขตอันตรายโดยอัตโนมัติ
    """

    def __init__(self, width: float = 100.0, height: float = 100.0):
        self.width   = width
        self.height  = height
        self.hazards: list[Hazard] = []

    def add_hazard(self, h: Hazard):
        self.hazards.append(h)

    def total_danger(self, px: float, py: float) -> float:
        return min(sum(h.danger_at(px, py) for h in self.hazards), 1.0)

    def movement_cost(self, ax: float, ay: float,
                      bx: float, by: float) -> float:
        dist = math.hypot(bx - ax, by - ay)
        mid  = self.total_danger((ax + bx) / 2, (ay + by) / 2)
        return dist * (1.0 + 4.0 * mid)

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return (
            max(0.0, min(self.width,  x)),
            max(0.0, min(self.height, y)),
        )


# ======================================================================
# 4.  ALGORITHM 1 — HILL CLIMBING
# ======================================================================

class HillClimbing:
    """
    Hill Climbing (Local Search)
    ════════════════════════════
    จาก lecture3_optimization.pdf:
      "search algorithms that maintain a single node
       and searches by moving to a neighboring node"

    function HILL-CLIMB(problem):
        current = initial state of problem
        repeat:
            neighbor = highest valued neighbor of current
            if neighbor not better than current:
                return current          ← local minimum
            current = neighbor

    Variants ที่ implement:
      steepest      : เลือก neighbor ที่ดีที่สุด (guaranteed best step)
      stochastic    : สุ่มจาก neighbor ที่ดีกว่า (escape some local optima)
      random_restart: ทำซ้ำหลายรอบจากจุดสุ่ม (ดีที่สุดในทางปฏิบัติ)

    Objective Function (minimize):
      f(x,y) = movement_cost(pos, exit) + 20 x total_danger(pos)
    """

    DIRS_8 = [
        ( 1,  0), (-1,  0), ( 0,  1), ( 0, -1),
        ( 1,  1), (-1,  1), ( 1, -1), (-1, -1),
    ]

    def __init__(self, dmap: DisasterMap, step: float = 3.0):
        self.map  = dmap
        self.step = step

    def _obj(self, x: float, y: float, ex: float, ey: float) -> float:
        return self.map.movement_cost(x, y, ex, ey) + self.map.total_danger(x, y) * 20.0

    def _neighbors(self, x: float, y: float) -> list[tuple[float, float]]:
        s = self.step
        return [self.map.clamp(x + dx*s, y + dy*s) for dx, dy in self.DIRS_8]

    def _climb(self, sx: float, sy: float, ex: float, ey: float,
               variant: str = "steepest", max_iter: int = 500) -> tuple[list, float]:
        cx, cy   = sx, sy
        path     = [(cx, cy)]
        cur_val  = self._obj(cx, cy, ex, ey)

        for _ in range(max_iter):
            nbrs = self._neighbors(cx, cy)

            if variant == "stochastic":
                better = [(n, self._obj(*n, ex, ey)) for n in nbrs
                          if self._obj(*n, ex, ey) < cur_val]
                if not better:
                    break
                chosen, chosen_val = random.choice(better)
            else:   # steepest
                chosen     = min(nbrs, key=lambda p: self._obj(*p, ex, ey))
                chosen_val = self._obj(*chosen, ex, ey)
                if chosen_val >= cur_val:
                    break

            cx, cy  = chosen
            cur_val = chosen_val
            path.append((cx, cy))

        path.append((ex, ey))
        return path, round(cur_val, 2)

    def find_path(self, sx: float, sy: float, ex: float, ey: float,
                  variant: str = "steepest",
                  restarts: int = 6,
                  max_iter: int = 500) -> tuple[list, float]:
        """
        variant: "steepest" | "stochastic" | "random_restart"
        Returns: (path, cost)
        """
        if variant == "random_restart":
            best_path, best_val = self._climb(sx, sy, ex, ey, "steepest", max_iter)
            for _ in range(restarts - 1):
                rx = random.uniform(0, self.map.width)
                ry = random.uniform(0, self.map.height)
                p, v = self._climb(rx, ry, ex, ey, "steepest", max_iter)
                if v < best_val:
                    best_path, best_val = p, v
            return best_path, best_val

        return self._climb(sx, sy, ex, ey, variant, max_iter)


# ======================================================================
# 5.  ALGORITHM 2 — SIMULATED ANNEALING
# ======================================================================

class SimulatedAnnealing:
    """
    Simulated Annealing
    ═══════════════════
    จาก lecture3_optimization.pdf:
      "Early on, higher temperature: more likely to accept
       neighbors that are worse than current state"

    Pseudocode (lecture):
    ─────────────────────
    function SIMULATED-ANNEALING(problem, max):
        current = initial state
        for t = 1 to max:
            T = TEMPERATURE(t)
            neighbor = random neighbor of current
            dE = how much better neighbor is than current
            if dE > 0:
                current = neighbor
            with probability e^(dE/T) set current = neighbor
        return current

    Cooling Schedule: T(t) = T0 x cooling^t   (exponential)
    """

    def __init__(self, dmap: DisasterMap):
        self.map = dmap

    def _obj(self, x: float, y: float, ex: float, ey: float) -> float:
        return self.map.movement_cost(x, y, ex, ey) + self.map.total_danger(x, y) * 20.0

    def find_path(self, sx: float, sy: float, ex: float, ey: float,
                  max_iter: int = 2000,
                  initial_T: float = 80.0,
                  cooling: float = 0.995,
                  step_size: float = 4.0) -> tuple[list, float]:
        """
        Parameters
        ──────────
        initial_T  : อุณหภูมิเริ่มต้น (สูง = explore มาก)
        cooling    : อัตราลดอุณหภูมิต่อ step  (0 < cooling < 1)
        step_size  : ขนาดก้าวสูงสุดต่อ iteration

        Returns: (path, total_movement_cost)
        """
        cx, cy = sx, sy
        path   = [(cx, cy)]
        T      = initial_T

        for _ in range(max_iter):
            T *= cooling

            angle = random.uniform(0, 2 * math.pi)
            dist  = random.uniform(step_size * 0.5, step_size)
            nx, ny = self.map.clamp(
                cx + math.cos(angle) * dist,
                cy + math.sin(angle) * dist,
            )

            cur_score = self._obj(cx, cy, ex, ey)
            new_score = self._obj(nx, ny, ex, ey)
            delta_E   = new_score - cur_score      # >0 = แย่ลง (minimize)

            if delta_E < 0:
                # ดีกว่า — รับทันที
                cx, cy = nx, ny
                path.append((cx, cy))
            elif T > 1e-6 and random.random() < math.exp(-delta_E / T):
                # แย่กว่า — รับด้วย probability e^(-dE/T)
                cx, cy = nx, ny
                path.append((cx, cy))

        path.append((ex, ey))

        # คำนวณ total movement cost จริง (ไม่รวม penalty)
        total = sum(
            self.map.movement_cost(path[i][0], path[i][1],
                                   path[i+1][0], path[i+1][1])
            for i in range(len(path) - 1)
        )
        return path, round(total, 2)


# ======================================================================
# 6.  ALGORITHM 3 — A* SEARCH
# ======================================================================

class AStarSearch:
    """
    A* Search
    ═════════
    f(n) = g(n) + h(n)
      g(n) : ต้นทุนสะสมจากจุดเริ่มต้นถึง n  (movement_cost จริง)
      h(n) : heuristic = Euclidean distance ถึง exit  (admissible)

    ค้นหาบน grid (grid_step x grid_step)
    รับประกัน optimal path เมื่อ h ไม่ overestimate
    """

    DIRS_8 = [
        ( 1,  0), (-1,  0), ( 0,  1), ( 0, -1),
        ( 1,  1), (-1,  1), ( 1, -1), (-1, -1),
    ]

    def __init__(self, dmap: DisasterMap, grid_step: float = 4.0):
        self.map       = dmap
        self.grid_step = grid_step

    def _snap(self, x: float, y: float) -> tuple[int, int]:
        g = self.grid_step
        return (round(x / g), round(y / g))

    def _unsnap(self, gx: int, gy: int) -> tuple[float, float]:
        return (gx * self.grid_step, gy * self.grid_step)

    def _key(self, g: tuple) -> int:
        return g[0] * 100000 + g[1]

    def find_path(self, sx: float, sy: float,
                  ex: float, ey: float) -> tuple[list, float]:
        """Returns: (path, total_cost)  path = list of (x, y)"""
        sg = self._snap(sx, sy)
        eg = self._snap(ex, ey)

        def h(gpos):
            px, py = self._unsnap(*gpos)
            return math.hypot(px - ex, py - ey)

        open_heap: list = []
        heapq.heappush(open_heap, (h(sg), sg))
        came_from: dict = {}
        g_score:   dict = {self._key(sg): 0.0}
        visited:   set  = set()

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            ck = self._key(cur)
            if ck in visited:
                continue
            visited.add(ck)

            if cur == eg:
                break

            cx_r, cy_r = self._unsnap(*cur)
            for dx, dy in self.DIRS_8:
                nb = (cur[0] + dx, cur[1] + dy)
                nx_r, ny_r = self._unsnap(*nb)
                if nx_r < 0 or ny_r < 0 or nx_r > self.map.width or ny_r > self.map.height:
                    continue
                nk     = self._key(nb)
                move   = self.map.movement_cost(cx_r, cy_r, nx_r, ny_r)
                new_g  = g_score[ck] + move
                if nk not in g_score or new_g < g_score[nk]:
                    g_score[nk]   = new_g
                    came_from[nk] = (ck, cur)
                    heapq.heappush(open_heap, (new_g + h(nb), nb))

        # Reconstruct
        path_grids = []
        node = eg
        while self._key(node) in came_from:
            path_grids.append(node)
            _, node = came_from[self._key(node)]
        path_grids.append(sg)
        path_grids.reverse()

        path = [(sx, sy)] + [self._unsnap(*g) for g in path_grids[1:]] + [(ex, ey)]
        dedup = [path[0]]
        for p in path[1:]:
            if abs(p[0] - dedup[-1][0]) > 0.01 or abs(p[1] - dedup[-1][1]) > 0.01:
                dedup.append(p)

        total = sum(
            self.map.movement_cost(dedup[i][0], dedup[i][1],
                                   dedup[i+1][0], dedup[i+1][1])
            for i in range(len(dedup) - 1)
        )
        return dedup, round(total, 2)


# ======================================================================
# 7.  ALGORITHM 4 — GREEDY LP (Resource Allocation)
# ======================================================================

class ResourceAllocator:
    """
    Greedy Linear Programming — จัดสรรทรัพยากรฉุกเฉิน
    ════════════════════════════════════════════════════
    Objective  : maximize total urgency resolved
    Constraints:
        sum(medkits_given)       <= available_medkits
        sum(food_given)          <= available_food
        sum(rescue_teams_given)  <= available_rescue_teams

    Strategy: เรียง urgency_score สูงสุดก่อน แล้ว allocate แบบ greedy
    """

    def allocate(self, survivors: list[Survivor],
                 medkits: int, food: int,
                 rescue_teams: int) -> dict[str, dict]:
        """
        Returns {survivor_id: {"medkit": 0|1, "food": 0|1, "rescue_team": 0|1}}
        """
        sorted_s = sorted(survivors, key=lambda s: s.urgency_score, reverse=True)
        result   = {}

        for s in sorted_s:
            alloc = {"medkit": 0, "food": 0, "rescue_team": 0}

            # health < 50 → ต้องการยา/ปฐมพยาบาล
            if s.health < 50 and medkits > 0:
                alloc["medkit"] = 1
                medkits -= 1

            # supplies < 40 → ต้องการเสบียง
            if s.supplies < 40 and food > 0:
                alloc["food"] = 1
                food -= 1

            # mobility < 30 หรือ health < 30 → ต้องการทีมช่วยเหลือ
            if (s.mobility < 30 or s.health < 30) and rescue_teams > 0:
                alloc["rescue_team"] = 1
                rescue_teams -= 1

            result[s.survivor_id] = alloc

        return result


# ======================================================================
# 8.  MAIN AI ADVISOR
# ======================================================================

class DisasterSurvivalAI:
    """
    ระบบ AI หลัก
    ════════════
    ขั้นตอน:
      1. คำนวณ urgency ผู้รอดชีวิต
      2. รัน optimization algorithm
      3. วิเคราะห์ danger ตลอดเส้นทาง
      4. จัดสรรทรัพยากร (LP)
      5. รวบรวมคำแนะนำตามประเภทภัย
      6. สร้าง result dict
    """

    def __init__(self, dmap: DisasterMap):
        self.map       = dmap
        self.hc        = HillClimbing(dmap)
        self.sa        = SimulatedAnnealing(dmap)
        self.astar     = AStarSearch(dmap)
        self.allocator = ResourceAllocator()

    # ── เลือก algorithm ─────────────────────────────────────────────
    def _run_algorithm(self, sx, sy, ex, ey, algorithm) -> tuple[list, float, str]:
        dispatch = {
            "sa":         (lambda: self.sa.find_path(sx, sy, ex, ey),
                           "Simulated Annealing"),
            "hc":         (lambda: self.hc.find_path(sx, sy, ex, ey, "steepest"),
                           "Hill Climbing (Steepest)"),
            "hc_restart": (lambda: self.hc.find_path(sx, sy, ex, ey, "random_restart"),
                           "Hill Climbing (Random Restart)"),
            "astar":      (lambda: self.astar.find_path(sx, sy, ex, ey),
                           "A* Search"),
        }
        fn, name = dispatch.get(algorithm, dispatch["astar"])
        path, cost = fn()
        return path, cost, name

    # ── วิเคราะห์ danger ตลอดเส้นทาง ───────────────────────────────
    def _path_danger_stats(self, path: list) -> dict:
        if not path:
            return {"max": 0.0, "avg": 0.0, "high_danger_count": 0}
        dangers = [self.map.total_danger(x, y) for x, y in path]
        return {
            "max":               round(max(dangers), 3),
            "avg":               round(sum(dangers) / len(dangers), 3),
            "high_danger_count": sum(1 for d in dangers if d > 0.4),
        }

    # ── API หลัก ─────────────────────────────────────────────────────
    def analyze(self,
                survivors   : list[Survivor],
                exit_x      : float,
                exit_y      : float,
                algorithm   : str = "astar",
                medkits     : int = 3,
                food        : int = 3,
                rescue_teams: int = 2) -> dict:
        """
        Parameters
        ──────────
        survivors    : list[Survivor]
        exit_x/y     : พิกัดทางออก
        algorithm    : "sa" | "hc" | "hc_restart" | "astar"
        medkits/food/rescue_teams : ทรัพยากรที่มี

        Returns
        ───────
        dict ผลการ optimization ครบถ้วน
        """
        # 1. เรียงลำดับผู้รอดชีวิตตาม urgency
        sorted_survivors = sorted(survivors,
                                  key=lambda s: s.urgency_score, reverse=True)
        leader = sorted_survivors[0]

        # 2. หาเส้นทาง
        t0 = time.perf_counter()
        path, cost, algo_name = self._run_algorithm(
            leader.x, leader.y, exit_x, exit_y, algorithm
        )
        elapsed = time.perf_counter() - t0

        # 3. วิเคราะห์ danger
        danger_stats = self._path_danger_stats(path)

        # 4. จัดสรรทรัพยากร
        resource_alloc = self.allocator.allocate(
            survivors, medkits, food, rescue_teams
        )

        # 5. รวบรวมคำแนะนำ
        hazard_types = list({h.hazard_type for h in self.map.hazards})
        tips = [
            {"type": ht, "label": HAZARD_LABELS[ht], "tip": tip}
            for ht in hazard_types
            for tip in SURVIVAL_TIPS.get(ht, [])
        ]

        # 6. รวมผล
        return {
            "algorithm"       : algo_name,
            "elapsed_sec"     : round(elapsed, 4),
            "path"            : path,
            "path_cost"       : cost,
            "path_waypoints"  : len(path),
            "danger_stats"    : danger_stats,
            "exit"            : (exit_x, exit_y),
            "exit_danger"     : round(self.map.total_danger(exit_x, exit_y), 3),
            "hazard_types"    : hazard_types,
            "survival_tips"   : tips,
            "sorted_survivors": [
                {
                    "id"           : s.survivor_id,
                    "pos"          : (s.x, s.y),
                    "health"       : s.health,
                    "mobility"     : s.mobility,
                    "supplies"     : s.supplies,
                    "urgency"      : round(s.urgency_score, 1),
                    "status"       : s.status_label,
                    "danger_at_pos": round(self.map.total_danger(s.x, s.y), 3),
                    "resources"    : resource_alloc.get(s.survivor_id, {}),
                }
                for s in sorted_survivors
            ],
        }

    # ── Pretty Print ─────────────────────────────────────────────────
    def print_report(self, result: dict):
        W = 68
        SEP = "=" * W
        print(f"\n{SEP}")
        print("  DISASTER SURVIVAL AI -- OPTIMIZATION REPORT")
        print(SEP)

        print(f"\n  Algorithm     : {result['algorithm']}")
        print(f"  คำนวณใน       : {result['elapsed_sec']} วินาที")
        print(f"  Waypoints     : {result['path_waypoints']} จุด")
        print(f"  Path Cost     : {result['path_cost']:.2f}")

        ds = result["danger_stats"]
        ex, ey = result["exit"]
        print(f"\n  --- อันตรายตลอดเส้นทาง ---")
        print(f"  Max danger    : {ds['max']:.1%}")
        print(f"  Avg danger    : {ds['avg']:.1%}")
        print(f"  จุดอันตรายสูง  : {ds['high_danger_count']} จุด (>40%)")
        print(f"  ทางออก        : ({ex}, {ey})  danger = {result['exit_danger']:.1%}")

        print(f"\n  --- ลำดับผู้รอดชีวิต (urgency สูงสุดก่อน) ---")
        for i, s in enumerate(result["sorted_survivors"], 1):
            res_parts = [k for k, v in s["resources"].items() if v]
            res_str   = ", ".join(res_parts) if res_parts else "ไม่มีทรัพยากร"
            print(f"  {i}. [{s['urgency']:5.1f}]  {s['id']:<14}  "
                  f"HP:{s['health']:3d}  Mob:{s['mobility']:3d}  "
                  f"{s['status']:<30}  ทรัพยากร: {res_str}")

        print(f"\n  --- คำแนะนำการเอาชีวิตรอด ---")
        prev_type = None
        for tip in result["survival_tips"]:
            if tip["type"] != prev_type:
                print(f"\n  [{tip['label']}]")
                prev_type = tip["type"]
            print(f"    * {tip['tip']}")

        print(f"\n{SEP}\n")


# ======================================================================
# 9.  DEMO
# ======================================================================

def demo():
    """
    สาธิตการใช้งานครบทุก step และเปรียบเทียบ 4 algorithms
    """
    random.seed(2024)

    # ── 1. สร้างแผนที่ ──────────────────────────────────────────────
    dmap = DisasterMap(width=100, height=100)

    # ── 2. เพิ่มภัยพิบัติ ───────────────────────────────────────────
    dmap.add_hazard(Hazard("fire",       x=25, y=25, severity=8, radius=15))
    dmap.add_hazard(Hazard("flood",      x=60, y=20, severity=6, radius=18))
    dmap.add_hazard(Hazard("collapse",   x=10, y=70, severity=7, radius=12))
    dmap.add_hazard(Hazard("toxic",      x=75, y=60, severity=5, radius=14))
    dmap.add_hazard(Hazard("earthquake", x=50, y=50, severity=9, radius=10))

    # ── 3. ผู้รอดชีวิต ──────────────────────────────────────────────
    survivors = [
        Survivor("Survivor-A", x=20, y=30, health=70, mobility=90, supplies=50),
        Survivor("Survivor-B", x=55, y=15, health=30, mobility=20, supplies=10),
        Survivor("Survivor-C", x= 8, y=65, health=55, mobility=60, supplies=30),
        Survivor("Survivor-D", x=70, y=55, health=80, mobility=80, supplies=80),
        Survivor("Survivor-E", x=45, y=40, health=10, mobility=15, supplies= 5),
    ]

    EXIT_X, EXIT_Y = 90, 90

    # ── 4. สร้าง AI ─────────────────────────────────────────────────
    ai = DisasterSurvivalAI(dmap)

    # ── 5. รัน Simulated Annealing และพิมพ์ report ───────────────────
    print("Running Simulated Annealing...")
    result = ai.analyze(
        survivors    = survivors,
        exit_x       = EXIT_X,
        exit_y       = EXIT_Y,
        algorithm    = "sa",
        medkits      = 3,
        food         = 2,
        rescue_teams = 2,
    )
    ai.print_report(result)

    # ── 6. เปรียบเทียบ 4 algorithms ─────────────────────────────────
    W = 68
    print("=" * W)
    print("  เปรียบเทียบประสิทธิภาพ 4 Algorithms")
    print("=" * W)
    print(f"  {'Algorithm':<38} {'Cost':>8}  {'Waypts':>7}  {'Time(s)':>8}")
    print("  " + "-" * 65)
    for algo in ["sa", "hc", "hc_restart", "astar"]:
        r = ai.analyze(survivors, EXIT_X, EXIT_Y, algorithm=algo)
        print(f"  {r['algorithm']:<38} {r['path_cost']:>8.1f}  "
              f"{r['path_waypoints']:>7}  {r['elapsed_sec']:>8.4f}")
    print("=" * W)


# ======================================================================
# 10.  QUICK-USE HELPER
# ======================================================================

def build_and_run(
    hazards_input   : list[dict],
    survivors_input : list[dict],
    exit_x          : float,
    exit_y          : float,
    algorithm       : str = "astar",
    medkits         : int = 3,
    food            : int = 3,
    rescue_teams    : int = 2,
    print_report    : bool = True,
) -> dict:
    """
    Helper สำหรับเรียกใช้ง่ายๆ จาก code ภายนอก

    Parameters
    ──────────
    hazards_input:
        [{"type": "fire", "x": 30, "y": 30, "severity": 7, "radius": 12}, ...]
    survivors_input:
        [{"id": "คนA", "x": 10, "y": 10, "health": 60, "mobility": 80, "supplies": 50}, ...]

    Example
    ───────
    result = build_and_run(
        hazards_input=[
            {"type": "fire",  "x": 30, "y": 30, "severity": 8, "radius": 12},
            {"type": "flood", "x": 60, "y": 50, "severity": 6, "radius": 15},
        ],
        survivors_input=[
            {"id": "A", "x": 15, "y": 25, "health": 70, "mobility": 90, "supplies": 50},
            {"id": "B", "x": 40, "y": 45, "health": 20, "mobility": 15, "supplies": 10},
        ],
        exit_x=90, exit_y=90,
        algorithm="astar",
    )
    print(result["path_cost"])
    """
    dmap = DisasterMap()
    for h in hazards_input:
        dmap.add_hazard(Hazard(h["type"], h["x"], h["y"],
                               h["severity"], h["radius"]))

    survivors = [
        Survivor(s["id"], s["x"], s["y"],
                 s["health"], s["mobility"], s["supplies"])
        for s in survivors_input
    ]

    ai     = DisasterSurvivalAI(dmap)
    result = ai.analyze(survivors, exit_x, exit_y, algorithm,
                        medkits, food, rescue_teams)
    if print_report:
        ai.print_report(result)
    return result


# ======================================================================
if __name__ == "__main__":
    demo()