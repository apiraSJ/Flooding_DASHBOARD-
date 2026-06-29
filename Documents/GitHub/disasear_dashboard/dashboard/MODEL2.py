"""
Disaster Survival AI - Route Optimization Model
================================================
ระบบ AI สำหรับคำนวณเส้นทางอพยพและกลยุทธ์เอาตัวรอดจากภัยพิบัติ
ใช้เทคนิค Optimization 6 แบบ:

    1. Hill Climbing              - Local Search (Greedy)
    2. Simulated Annealing        - Local Search (Probabilistic Escape)
    3. Genetic Algorithm          - Evolutionary Optimization
    4. Backtracking Search        - CSP / Exhaustive Search
    5. Arc Consistency (AC-3)     - CSP with Constraint Propagation
    6. Linear Programming (LP)    - Mathematical Optimization (Survival Advice)

Author  : DisasterSurvivalAI
Requires: scipy
"""

import math
import random
import itertools
from scipy.optimize import linprog


# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def euclidean(p1, p2):
    """ระยะห่างแบบ Euclidean ระหว่างสองจุด"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def zone_cost(user_loc, zone, disaster_loc, severity):
    """
    ฟังก์ชัน Cost สำหรับการประเมิน Safe Zone
    - ยิ่งใกล้ zone ยิ่งดี (dist_to_safe ต่ำ)
    - ยิ่ง zone อยู่ใกล้ภัย ยิ่งแย่ (danger_penalty สูง)
    - ยิ่งเส้นทาง user→zone ผ่านใกล้ภัย ยิ่งแย่ (path_danger สูง)
    """
    dist_to_safe      = euclidean(user_loc, zone)
    dist_zone_dis     = euclidean(zone, disaster_loc)
    dist_user_dis     = euclidean(user_loc, disaster_loc)

    danger_penalty    = (severity * 10) / (dist_zone_dis + 0.1)
    path_danger       = (severity * 5)  / (dist_user_dis + 0.1)

    return dist_to_safe + danger_penalty + path_danger


# ─────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────────

class DisasterSurvivalAI:
    """
    ระบบ AI วิเคราะห์และแนะนำเส้นทางอพยพจากภัยพิบัติ

    Attributes:
        user_loc      (tuple): พิกัดผู้ใช้ (x, y)
        disaster_loc  (tuple): พิกัดจุดศูนย์กลางภัยพิบัติ (x, y)
        severity      (int)  : ระดับความรุนแรง 1-5 (5 = รุนแรงสูงสุด)
        safe_zones    (list) : รายการพิกัด Safe Zone [(x1,y1), ...]
    """

    def __init__(self, user_loc, disaster_loc, severity, safe_zones):
        self.user_loc     = user_loc
        self.disaster_loc = disaster_loc
        self.severity     = max(1, min(5, severity))   # clamp 1-5
        self.safe_zones   = safe_zones

    # ──────────────────────────────────────────────────────────────
    #  1. HILL CLIMBING
    # ──────────────────────────────────────────────────────────────
    def optimize_hill_climbing(self):
        """
        Hill Climbing — Local Search แบบ Greedy

        เริ่มจาก zone สุ่มหนึ่งจุด แล้วมองไปยัง "เพื่อนบ้าน" ที่ใกล้เคียง
        (ในที่นี้คือ zone ที่เหลือทั้งหมด) เลือกเฉพาะตัวที่ดีกว่าเสมอ
        หาก iteration ไหนไม่มีตัวดีกว่า = ติด Local Minimum

        Returns:
            dict: {zone, cost, path, status}
        """
        if not self.safe_zones:
            return {"zone": None, "cost": float("inf"), "status": "ไม่มี Safe Zone"}

        # เริ่มจาก zone แรก
        current      = self.safe_zones[0]
        current_cost = zone_cost(self.user_loc, current,
                                 self.disaster_loc, self.severity)
        path         = [current]
        iterations   = 0
        stuck        = False

        for _ in range(len(self.safe_zones) * 2):
            improved = False
            for zone in self.safe_zones:
                if zone == current:
                    continue
                c = zone_cost(self.user_loc, zone, self.disaster_loc, self.severity)
                if c < current_cost:
                    current      = zone
                    current_cost = c
                    path.append(current)
                    improved     = True
                    break          # เลือกตัวแรกที่ดีกว่า (Steepest-Ascent variant)

            iterations += 1
            if not improved:
                stuck = True
                break

        status = (
            f"[Hill Climbing] ติด Local Minimum หลัง {iterations} รอบ!"
            if stuck else
            f"[Hill Climbing] บรรจบใน {iterations} รอบ"
        )
        return {"zone": current, "cost": current_cost,
                "path": path, "status": status, "iterations": iterations}

    # ──────────────────────────────────────────────────────────────
    #  2. SIMULATED ANNEALING
    # ──────────────────────────────────────────────────────────────
    def optimize_simulated_annealing(self, T_init=100.0, T_min=0.1, alpha=0.90,
                                     max_iter=500):
        """
        Simulated Annealing — หลบหลีก Local Minimum ด้วย Probabilistic Acceptance

        ยอมรับ solution ที่แย่กว่าได้ตามโอกาส exp(−ΔE / T)
        Temperature ลดลงตาม Cooling Schedule: T ← T × alpha

        Args:
            T_init   : อุณหภูมิเริ่มต้น
            T_min    : อุณหภูมิต่ำสุด (หยุดเมื่อถึงค่านี้)
            alpha    : อัตราการลดอุณหภูมิ (0 < alpha < 1)
            max_iter : จำนวน iteration สูงสุด

        Returns:
            dict: {zone, cost, T_final, accepted_worse, status}
        """
        if not self.safe_zones:
            return {"zone": None, "cost": float("inf"), "status": "ไม่มี Safe Zone"}

        current       = random.choice(self.safe_zones)
        current_cost  = zone_cost(self.user_loc, current,
                                  self.disaster_loc, self.severity)
        best          = current
        best_cost     = current_cost
        T             = T_init
        accepted_worse = 0

        for i in range(max_iter):
            if T < T_min:
                break

            candidate      = random.choice(self.safe_zones)
            candidate_cost = zone_cost(self.user_loc, candidate,
                                       self.disaster_loc, self.severity)
            delta          = candidate_cost - current_cost   # ΔE

            if delta < 0:
                # ดีกว่า — รับทันที
                current      = candidate
                current_cost = candidate_cost
            else:
                # แย่กว่า — รับตาม Boltzmann probability
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current       = candidate
                    current_cost  = candidate_cost
                    accepted_worse += 1

            if current_cost < best_cost:
                best      = current
                best_cost = current_cost

            T *= alpha

        status = (
            f"[Simulated Annealing] T: {T_init}→{T:.4f} | "
            f"ยอมรับ worse move {accepted_worse} ครั้ง | "
            f"Best cost: {best_cost:.2f}"
        )
        return {"zone": best, "cost": best_cost,
                "T_final": T, "accepted_worse": accepted_worse,
                "status": status}

    # ──────────────────────────────────────────────────────────────
    #  3. GENETIC ALGORITHM
    # ──────────────────────────────────────────────────────────────
    def optimize_genetic_algorithm(self, pop_size=30, generations=50,
                                   mutation_rate=0.2, elite_ratio=0.2):
        """
        Genetic Algorithm — วิวัฒนาการประชากรเพื่อค้นหา Safe Zone ที่ดีที่สุด

        แต่ละ Individual คือ sequence ของ Safe Zones (chromosome)
        Fitness = −cost (ยิ่งต่ำยิ่งดี → fitness สูง)
        Crossover = One-Point Crossover บน index ของ zones
        Mutation  = สุ่มเปลี่ยน gene เป็น zone อื่น

        Args:
            pop_size     : ขนาดประชากร
            generations  : จำนวน generation
            mutation_rate: ความน่าจะเป็นของ mutation
            elite_ratio  : สัดส่วนของ elite ที่รอดไปรุ่นถัดไป

        Returns:
            dict: {zone, cost, best_generation, status}
        """
        if not self.safe_zones:
            return {"zone": None, "cost": float("inf"), "status": "ไม่มี Safe Zone"}

        n = len(self.safe_zones)

        def fitness(zone):
            return -zone_cost(self.user_loc, zone, self.disaster_loc, self.severity)

        # Initialize population
        population = [random.choice(self.safe_zones) for _ in range(pop_size)]

        best_zone       = None
        best_cost       = float("inf")
        best_gen        = 0

        for gen in range(generations):
            # Evaluate
            scored = sorted(population, key=fitness, reverse=True)

            # Track global best
            gen_best      = scored[0]
            gen_best_cost = zone_cost(self.user_loc, gen_best,
                                      self.disaster_loc, self.severity)
            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_zone = gen_best
                best_gen  = gen + 1

            # Elitism
            elite_count = max(1, int(pop_size * elite_ratio))
            new_pop = scored[:elite_count]

            # Crossover + Mutation (ทำงานบน index ของ safe_zones)
            indices = list(range(n))
            scored_idx = sorted(indices,
                                 key=lambda i: fitness(self.safe_zones[i]),
                                 reverse=True)
            parent_pool = [self.safe_zones[i]
                           for i in scored_idx[:max(2, n // 2)]]

            while len(new_pop) < pop_size:
                p1, p2 = random.sample(parent_pool, min(2, len(parent_pool)))
                # One-point crossover (เลือกพ่อหรือแม่)
                child = p1 if random.random() < 0.5 else p2
                # Mutation
                if random.random() < mutation_rate:
                    child = random.choice(self.safe_zones)
                new_pop.append(child)

            population = new_pop

        status = (
            f"[Genetic Algorithm] {generations} generations | "
            f"Best found gen {best_gen} | "
            f"Best cost: {best_cost:.2f}"
        )
        return {"zone": best_zone, "cost": best_cost,
                "best_generation": best_gen, "status": status}

    # ──────────────────────────────────────────────────────────────
    #  4. BACKTRACKING SEARCH
    # ──────────────────────────────────────────────────────────────
    def optimize_backtracking(self):
        """
        Backtracking Search — ค้นหาแบบ Exhaustive พร้อม Pruning

        มองปัญหาเป็น CSP:
            Variables   = ลำดับ Safe Zone ที่จะไป (permutation)
            Constraints = cost ของ zone ต้องต่ำกว่า threshold
                          (danger_penalty ต้องไม่เกิน severity * 15)

        Backtrack ทันทีที่ constraint ถูกละเมิด

        Returns:
            dict: {zone, cost, nodes_explored, backtracks, status}
        """
        if not self.safe_zones:
            return {"zone": None, "cost": float("inf"), "status": "ไม่มี Safe Zone"}

        best          = {"zone": None, "cost": float("inf")}
        nodes         = [0]
        backtracks    = [0]
        danger_limit  = self.severity * 15   # constraint: danger_penalty ต้องไม่เกิน

        def is_feasible(zone):
            """Constraint check — ใช้สำหรับ pruning"""
            dist_zone_dis  = euclidean(zone, self.disaster_loc)
            danger_penalty = (self.severity * 10) / (dist_zone_dis + 0.1)
            return danger_penalty <= danger_limit

        def backtrack(remaining, current_path):
            nodes[0] += 1

            # Base case: ลองทุก zone แล้ว
            if not remaining:
                return

            for i, zone in enumerate(remaining):
                # ── Constraint check (Pruning) ──
                if not is_feasible(zone):
                    backtracks[0] += 1
                    continue           # Backtrack ทันที

                c = zone_cost(self.user_loc, zone, self.disaster_loc, self.severity)
                if c < best["cost"]:
                    best["cost"] = c
                    best["zone"] = zone

                next_remaining = remaining[:i] + remaining[i+1:]
                backtrack(next_remaining, current_path + [zone])

        backtrack(self.safe_zones, [])

        status = (
            f"[Backtracking] Explored {nodes[0]} nodes | "
            f"Backtracks: {backtracks[0]} | "
            f"Best cost: {best['cost']:.2f}"
        )
        return {"zone": best["zone"], "cost": best["cost"],
                "nodes_explored": nodes[0], "backtracks": backtracks[0],
                "status": status}

    # ──────────────────────────────────────────────────────────────
    #  5. ARC CONSISTENCY (AC-3 inspired)
    # ──────────────────────────────────────────────────────────────
    def optimize_arc_consistency(self):
        """
        Arc Consistency (AC-3 Inspired) — Constraint Propagation

        จำลอง AC-3 บนปัญหาการเลือก Safe Zone:
            Domain     = safe_zones ทั้งหมด
            Variable X = Safe Zone ที่จะเลือก
            Arc (X→Y)  = X ดีกว่า Y เสมอ ถ้า cost(X) < cost(Y)

        ขั้นตอน:
            1. ลด Domain โดย prune zones ที่ "dominated" (มี zone อื่นดีกว่าทุกด้าน)
            2. ลด Domain ต่อโดย prune zones ที่ danger_penalty เกิน threshold
            3. เลือก Best จาก Domain ที่เหลือ

        Returns:
            dict: {zone, cost, domain_before, domain_after, pruned, status}
        """
        if not self.safe_zones:
            return {"zone": None, "cost": float("inf"), "status": "ไม่มี Safe Zone"}

        domain_before  = list(self.safe_zones)
        domain         = list(self.safe_zones)
        pruned_zones   = []

        # ── Step 1: Constraint Propagation — danger threshold ──
        danger_threshold = self.severity * 12
        filtered = []
        for zone in domain:
            dist_z = euclidean(zone, self.disaster_loc)
            dp     = (self.severity * 10) / (dist_z + 0.1)
            if dp <= danger_threshold:
                filtered.append(zone)
            else:
                pruned_zones.append(("danger_threshold", zone))

        domain = filtered if filtered else domain  # fallback ถ้า prune หมด

        # ── Step 2: Arc Consistency — Dominance Pruning ──
        # prune zone ที่ "dominated" — มี zone อื่นที่ดีกว่าทั้ง dist_to_safe และ dist_from_disaster
        non_dominated = []
        for z in domain:
            dominated = False
            dz_safe = euclidean(self.user_loc, z)
            dz_dis  = euclidean(z, self.disaster_loc)
            for other in domain:
                if other == z:
                    continue
                do_safe = euclidean(self.user_loc, other)
                do_dis  = euclidean(other, self.disaster_loc)
                if do_safe <= dz_safe and do_dis >= dz_dis:
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(z)
            else:
                pruned_zones.append(("dominated", z))

        domain = non_dominated if non_dominated else domain  # fallback

        # ── Step 3: เลือก Best จาก reduced domain ──
        best_zone = min(domain,
                        key=lambda z: zone_cost(self.user_loc, z,
                                                self.disaster_loc, self.severity))
        best_cost = zone_cost(self.user_loc, best_zone,
                              self.disaster_loc, self.severity)

        status = (
            f"[AC-3] Domain {len(domain_before)}→{len(domain)} zones | "
            f"Pruned {len(pruned_zones)} | "
            f"Best cost: {best_cost:.2f}"
        )
        return {
            "zone"         : best_zone,
            "cost"         : best_cost,
            "domain_before": domain_before,
            "domain_after" : domain,
            "pruned"       : pruned_zones,
            "status"       : status,
        }

    # ──────────────────────────────────────────────────────────────
    #  6. LINEAR PROGRAMMING — Survival Advice
    # ──────────────────────────────────────────────────────────────
    def optimize_survival_advice_lp(self):
        """
        Linear Programming — จัดสรรเวลาทำกิจกรรมเอาตัวรอดให้เหมาะสมที่สุด

        Variables  x = [x1, x2, x3] ∈ [0,1]
            x1 = สัดส่วนเวลาที่ทุ่มกับการอพยพ         (ใช้ 10 นาที/หน่วย)
            x2 = สัดส่วนเวลาที่ทุ่มกับการเตรียมเสบียง  (ใช้ 20 นาที/หน่วย)
            x3 = สัดส่วนเวลาที่ทุ่มกับการหาที่หลบซ่อน  (ใช้ 5  นาที/หน่วย)

        Objective  : Maximize  Σ utility_i × x_i
        Constraint : Σ time_i  × x_i  ≤  time_limit (นาที)

        Utility ปรับตาม severity แบบ continuous (ไม่ใช่ if/else แบบเดิม)

        Returns:
            dict: {advice, x_values, utility, time_used, status}
        """
        s = self.severity   # 1-5

        # Utility ของแต่ละกิจกรรม ปรับตาม severity
        u_evac    = 2 * s                      # ยิ่งรุนแรง อพยพสำคัญมาก
        u_supply  = max(1.0, (6 - s) * 1.5)   # ยิ่งรุนแรงน้อย เตรียมของสำคัญขึ้น
        u_shelter = s * 0.8                    # หลบซ่อนสำคัญตามความรุนแรง

        # เวลาที่มีจำกัด (นาที) — ยิ่งรุนแรงยิ่งเร่งด่วน
        time_limit = max(10, 50 - s * 7)      # severity=5 → 15 นาที, severity=1 → 43 นาที

        # Minimize −utility (linprog ทำ Minimize)
        c      = [-u_evac, -u_supply, -u_shelter]
        A_ub   = [[10, 20, 5]]
        b_ub   = [time_limit]
        bounds = [(0, 1), (0, 1), (0, 1)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        advice_pool = [
            f"⚠️  [อพยพทันที x={res.x[0]:.2f}] มุ่งหน้าตามเส้นทางที่ระบบคำนวณ ห้ามรีรอ!",
            f"🎒 [เตรียมเสบียง x={res.x[1]:.2f}] หยิบน้ำดื่ม ยา และเอกสารสำคัญ",
            f"🧱 [หาที่กำบัง x={res.x[2]:.2f}] หลบในโครงสร้างตึกที่แข็งแรงที่สุด",
        ]

        THRESHOLD = 0.25
        selected  = [advice_pool[i] for i, v in enumerate(res.x) if v > THRESHOLD]

        # Fallback — ต้องมีคำแนะนำอย่างน้อย 1 ข้อเสมอ
        if not selected:
            selected = [advice_pool[0]]

        time_used = sum(t * x for t, x in zip([10, 20, 5], res.x))
        utility   = -res.fun

        status = (
            f"[LP] severity={s} | time_limit={time_limit} นาที | "
            f"Time used={time_used:.1f} นาที | "
            f"Total utility={utility:.2f} | "
            f"x=[{res.x[0]:.2f}, {res.x[1]:.2f}, {res.x[2]:.2f}]"
        )
        return {
            "advice"    : selected,
            "x_values"  : res.x.tolist(),
            "utility"   : utility,
            "time_used" : time_used,
            "time_limit": time_limit,
            "status"    : status,
        }

    # ──────────────────────────────────────────────────────────────
    #  RUN ALL — เรียกทุก method แล้วสรุปผล
    # ──────────────────────────────────────────────────────────────
    def run_all(self):
        """รันทุก optimization method แล้วสรุปผลเปรียบเทียบ"""
        print("=" * 65)
        print(f"  RescuOpt AI — Disaster Survival Optimizer")
        print(f"  User: {self.user_loc} | Disaster: {self.disaster_loc} | "
              f"Severity: {self.severity}/5")
        print("=" * 65)

        results = {}

        # ── Route Optimization (Method 1-5) ──
        print("\n📍 ROUTE OPTIMIZATION — หา Safe Zone ที่เหมาะสมที่สุด\n")
        route_methods = [
            ("1. Hill Climbing",          self.optimize_hill_climbing),
            ("2. Simulated Annealing",     self.optimize_simulated_annealing),
            ("3. Genetic Algorithm",       self.optimize_genetic_algorithm),
            ("4. Backtracking Search",     self.optimize_backtracking),
            ("5. Arc Consistency (AC-3)",  self.optimize_arc_consistency),
        ]

        for name, method in route_methods:
            res = method()
            results[name] = res
            print(f"  {name}")
            print(f"    Zone : {res['zone']}")
            print(f"    Cost : {res['cost']:.4f}")
            print(f"    {res['status']}")
            print()

        # ── Survival Advice (Method 6) ──
        print("─" * 65)
        print("\n📋 SURVIVAL ADVICE — Linear Programming (Method 6)\n")
        lp = self.optimize_survival_advice_lp()
        results["6. Linear Programming"] = lp
        print(f"  {lp['status']}\n")
        for adv in lp["advice"]:
            print(f"    {adv}")
        print()

        # ── Summary ──
        print("─" * 65)
        print("\n🏆 SUMMARY — เปรียบเทียบผล Route Optimization\n")
        print(f"  {'Method':<30} {'Zone':>12}  {'Cost':>10}")
        print(f"  {'─'*30} {'─'*12}  {'─'*10}")
        for name, res in results.items():
            if "zone" in res and res["zone"] is not None:
                print(f"  {name:<30} {str(res['zone']):>12}  {res['cost']:>10.4f}")

        # Best zone across all methods
        route_results = [(n, r) for n, r in results.items() if "zone" in r and r["zone"]]
        if route_results:
            best_name, best_res = min(route_results, key=lambda x: x[1]["cost"])
            print(f"\n  ✅ สรุป: วิธีที่ให้ผลดีที่สุด = {best_name}")
            print(f"         → Safe Zone: {best_res['zone']}  "
                  f"(Cost: {best_res['cost']:.4f})")
        print()
        print("=" * 65)

        return results


# ─────────────────────────────────────────────────────────────────
#  EXECUTION
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    ai = DisasterSurvivalAI(
        user_loc     = (0, 0),
        disaster_loc = (3, 3),
        severity     = 5,
        safe_zones   = [(1, 5), (4, 4), (10, 2), (6, 8), (2, 9)],
    )

    ai.run_all()