
"""
CS50 AI Lecture 3 - Optimization & Pathfinding Visualizer
==========================================================
Algorithms: A* Search, Hill Climbing, Simulated Annealing,
            Genetic Algorithm, Backtracking Search, Arc Consistency (AC-3)

Features:
  - Cost labels (g, h, f) rendered inside each cell
  - Pause / Resume button
  - Step Back / Step Forward buttons
  - Auto-generated random maze button
"""

import tkinter as tk
from tkinter import ttk
import random
import math
import collections
import heapq


# ──────────────────────────────────────────────
#  CONSTANTS & COLOUR PALETTE
# ──────────────────────────────────────────────
GRID_SIZE   = 15
CELL_PX     = 42
PAD         = 3

C_BG        = "#1a1a2e"
C_PANEL     = "#16213e"
C_ACCENT    = "#0f3460"
C_GRID_LINE = "#0d2137"

C_EMPTY     = "#0d2137"
C_WALL      = "#e94560"
C_START     = "#4ecca3"
C_GOAL      = "#f9ca24"
C_VISITED   = "#1a4a6e"
C_FRONTIER  = "#2980b9"
C_PATH      = "#a29bfe"
C_CURRENT   = "#fd79a8"
C_PRUNED    = "#2d1b2e"
C_GA_POP    = "#fdcb6e"
C_GA_BEST   = "#00b894"
C_BACKTRACK = "#e17055"
C_TEXT_MAIN = "#eaeaea"
C_TEXT_DIM  = "#74b9ff"
C_COST_TXT  = "#ffffff"

EMPTY = 0; WALL = 1; START = 2; GOAL = 3

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def neighbors(grid, r, c):
    result = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr][nc] != WALL:
            result.append((nr, nc))
    return result

def in_bounds(r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE

def is_passable(grid, r, c):
    return in_bounds(r, c) and grid[r][c] != WALL


# ──────────────────────────────────────────────
#  MAIN APPLICATION
# ──────────────────────────────────────────────
class PathfindingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CS50 AI — Optimization & Pathfinding Visualizer")
        self.root.configure(bg=C_BG)
        self.root.resizable(False, False)

        self.grid       = [[EMPTY] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.start      = None
        self.goal       = None
        self.place_mode = tk.StringVar(value="wall")

        # Animation state
        self.running    = False
        self.paused     = False
        self.after_id   = None
        self.anim_steps = []
        self.anim_index = 0
        self.overlay    = {}
        self.cost_map   = {}
        self.last_algo  = None

        self._build_ui()
        self._draw_grid()

    # ── UI CONSTRUCTION ──────────────────────────────────
    def _build_ui(self):
        canvas_frame = tk.Frame(self.root, bg=C_BG)
        canvas_frame.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="n")

        canvas_w = GRID_SIZE * CELL_PX + 2
        canvas_h = GRID_SIZE * CELL_PX + 2
        self.canvas = tk.Canvas(canvas_frame, width=canvas_w, height=canvas_h,
                                bg=C_GRID_LINE, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>",  self._on_left_click)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<Button-3>",  self._on_right_click)

        ctrl = tk.Frame(self.root, bg=C_PANEL, width=310)
        ctrl.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="ns")
        ctrl.grid_propagate(False)

        pad = dict(padx=12, pady=4)

        # Title
        tk.Label(ctrl, text="CS50 AI", bg=C_PANEL, fg=C_ACCENT,
                 font=("Helvetica", 11, "bold")).pack(**pad, anchor="w")
        tk.Label(ctrl, text="Optimization Visualizer", bg=C_PANEL, fg=C_TEXT_MAIN,
                 font=("Helvetica", 14, "bold")).pack(padx=12, pady=(0, 2), anchor="w")
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Place Mode ──
        tk.Label(ctrl, text="PLACE MODE", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        mode_frame = tk.Frame(ctrl, bg=C_PANEL)
        mode_frame.pack(padx=12, pady=2, fill="x")
        for text, val, color in [("Start (S)", "start", C_START),
                                   ("Goal (G)",  "goal",  C_GOAL),
                                   ("Wall (X)",  "wall",  C_WALL)]:
            tk.Radiobutton(mode_frame, text=text, variable=self.place_mode,
                           value=val, bg=C_PANEL, fg=color, selectcolor=C_ACCENT,
                           activebackground=C_PANEL, activeforeground=color,
                           font=("Helvetica", 10, "bold"), cursor="hand2"
                           ).pack(anchor="w", pady=1)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Algorithm ──
        tk.Label(ctrl, text="ALGORITHM", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        self.algo_var = tk.StringVar(value="A* Search")
        ttk.Combobox(ctrl, textvariable=self.algo_var, state="readonly",
                     font=("Helvetica", 10),
                     values=["A* Search",
                             "Hill Climbing",
                             "Simulated Annealing",
                             "Genetic Algorithm",
                             "Backtracking Search",
                             "Arc Consistency (AC-3)"]
                     ).pack(padx=12, pady=4, fill="x")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Speed ──
        tk.Label(ctrl, text="ANIMATION SPEED", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        spd_frame = tk.Frame(ctrl, bg=C_PANEL)
        spd_frame.pack(padx=12, pady=2, fill="x")
        tk.Label(spd_frame, text="Fast", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 8)).pack(side="left")
        self.speed_var = tk.IntVar(value=150)
        tk.Scale(spd_frame, variable=self.speed_var, from_=20, to=800,
                 orient="horizontal", bg=C_PANEL, fg=C_TEXT_MAIN,
                 troughcolor=C_ACCENT, highlightthickness=0,
                 showvalue=False).pack(side="left", fill="x", expand=True)
        tk.Label(spd_frame, text="Slow", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 8)).pack(side="left")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Primary Buttons ──
        btn_cfg = dict(font=("Helvetica", 11, "bold"), bd=0, cursor="hand2",
                       pady=7, relief="flat")

        self.run_btn = tk.Button(ctrl, text="▶  RUN",
                                 bg="#00b894", fg="white",
                                 command=self._run, **btn_cfg)
        self.run_btn.pack(padx=12, pady=3, fill="x")

        self.pause_btn = tk.Button(ctrl, text="⏸  PAUSE",
                                   bg="#6c5ce7", fg="white",
                                   command=self._toggle_pause,
                                   state="disabled", **btn_cfg)
        self.pause_btn.pack(padx=12, pady=3, fill="x")

        self.stop_btn = tk.Button(ctrl, text="■  STOP",
                                  bg="#d63031", fg="white",
                                  command=self._stop,
                                  state="disabled", **btn_cfg)
        self.stop_btn.pack(padx=12, pady=3, fill="x")

        # ── Step Navigation ──
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)
        tk.Label(ctrl, text="STEP NAVIGATION", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")

        nav_frame = tk.Frame(ctrl, bg=C_PANEL)
        nav_frame.pack(padx=12, pady=3, fill="x")
        nav_btn = dict(font=("Helvetica", 10, "bold"), bd=0, cursor="hand2",
                       pady=6, relief="flat", bg=C_ACCENT, fg=C_TEXT_MAIN)
        self.prev_btn = tk.Button(nav_frame, text="◀  PREV",
                                  command=self._step_back,
                                  state="disabled", **nav_btn)
        self.prev_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.next_btn = tk.Button(nav_frame, text="NEXT  ▶",
                                  command=self._step_forward,
                                  state="disabled", **nav_btn)
        self.next_btn.pack(side="left", expand=True, fill="x")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Utility Buttons ──
        tk.Button(ctrl, text="🎲  RANDOM MAZE",
                  bg="#fdcb6e", fg="#2d3436",
                  command=self._random_maze, **btn_cfg
                  ).pack(padx=12, pady=3, fill="x")
        tk.Button(ctrl, text="↺  RESET GRID",
                  bg=C_ACCENT, fg=C_TEXT_MAIN,
                  command=self._reset, **btn_cfg
                  ).pack(padx=12, pady=3, fill="x")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Legend ──
        tk.Label(ctrl, text="LEGEND", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        legend_items = [
            (C_START,    "Start"),
            (C_GOAL,     "Goal"),
            (C_WALL,     "Wall"),
            (C_VISITED,  "Visited"),
            (C_CURRENT,  "Current"),
            (C_PATH,     "Final Path"),
            (C_FRONTIER, "Frontier"),
            (C_BACKTRACK,"Backtrack"),
            (C_GA_POP,   "GA Pop"),
            (C_GA_BEST,  "GA Best"),
            (C_PRUNED,   "Pruned"),
        ]
        leg_frame = tk.Frame(ctrl, bg=C_PANEL)
        leg_frame.pack(padx=12, pady=2, fill="x")
        for i, (color, label) in enumerate(legend_items):
            row_f = tk.Frame(leg_frame, bg=C_PANEL)
            row_f.grid(row=i // 2, column=i % 2, sticky="w", padx=2, pady=1)
            tk.Label(row_f, bg=color, width=2, height=1).pack(side="left", padx=(0, 4))
            tk.Label(row_f, text=label, bg=C_PANEL, fg=C_TEXT_MAIN,
                     font=("Helvetica", 8)).pack(side="left")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)

        # ── Status & Step Counter ──
        tk.Label(ctrl, text="STATUS", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        self.status_var = tk.StringVar(value="Ready. Place Start (S) and Goal (G), add walls, then RUN.")
        tk.Label(ctrl, textvariable=self.status_var,
                 bg=C_ACCENT, fg=C_TEXT_MAIN,
                 font=("Helvetica", 9), wraplength=270,
                 justify="left", anchor="nw"
                 ).pack(padx=12, pady=4, fill="x", ipadx=8, ipady=8)

        self.step_var = tk.StringVar(value="Step: —")
        tk.Label(ctrl, textvariable=self.step_var, bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9)).pack(padx=12, anchor="w")

        # ── Results ──
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=5)
        tk.Label(ctrl, text="RESULTS", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")

        results_box = tk.Frame(ctrl, bg=C_ACCENT)
        results_box.pack(padx=12, pady=4, fill="x")

        self.cost_var  = tk.StringVar(value="Total Cost Used: —")
        self.dist_var  = tk.StringVar(value="Distance Traveled: —")
        self.optim_var = tk.StringVar(value="")

        # FIX: use ipadx/ipady on pack() instead of padx/pady tuples in Label()
        lbl_cost = tk.Label(results_box, textvariable=self.cost_var,
                            bg=C_ACCENT, fg=C_TEXT_MAIN,
                            font=("Helvetica", 10, "bold"),
                            anchor="w", justify="left")
        lbl_cost.pack(fill="x", padx=8, pady=(8, 2))

        lbl_dist = tk.Label(results_box, textvariable=self.dist_var,
                            bg=C_ACCENT, fg=C_TEXT_MAIN,
                            font=("Helvetica", 9),
                            anchor="w", justify="left", wraplength=270)
        lbl_dist.pack(fill="x", padx=8, pady=2)

        lbl_optim = tk.Label(results_box, textvariable=self.optim_var,
                             bg=C_ACCENT, fg="#a8e6cf",
                             font=("Helvetica", 8, "bold"),
                             anchor="w", justify="left", wraplength=270)
        lbl_optim.pack(fill="x", padx=8, pady=(2, 8))

    # ── GRID DRAWING ─────────────────────────────────────
    def _cell_coords(self, r, c):
        x0 = c * CELL_PX + PAD
        y0 = r * CELL_PX + PAD
        x1 = x0 + CELL_PX - PAD * 2
        y1 = y0 + CELL_PX - PAD * 2
        return x0, y0, x1, y1

    def _draw_grid(self):
        self.canvas.delete("all")
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self._draw_cell(r, c)

    def _cell_color(self, r, c):
        s = self.grid[r][c]
        if s == START: return C_START
        if s == GOAL:  return C_GOAL
        if s == WALL:  return C_WALL
        if (r, c) in self.overlay:
            return self.overlay[(r, c)]
        return C_EMPTY

    def _draw_cell(self, r, c):
        x0, y0, x1, y1 = self._cell_coords(r, c)
        color = self._cell_color(r, c)
        tag = f"cell_{r}_{c}"
        self.canvas.delete(tag)
        self.canvas.create_rectangle(x0, y0, x1, y1,
                                     fill=color, outline=C_GRID_LINE,
                                     width=1, tags=tag)
        state = self.grid[r][c]
        if state == START:
            self.canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2,
                                    text="S", fill=C_BG,
                                    font=("Helvetica", int(CELL_PX * 0.38), "bold"),
                                    tags=tag)
        elif state == GOAL:
            self.canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2,
                                    text="G", fill=C_BG,
                                    font=("Helvetica", int(CELL_PX * 0.38), "bold"),
                                    tags=tag)
        elif state == WALL:
            self.canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2,
                                    text="✕", fill="white",
                                    font=("Helvetica", int(CELL_PX * 0.32), "bold"),
                                    tags=tag)
        elif (r, c) in self.cost_map:
            label = self.cost_map[(r, c)]
            lines = str(label).split("\n")
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            if len(lines) == 1:
                self.canvas.create_text(cx, cy,
                                        text=lines[0],
                                        fill=C_TEXT_MAIN,
                                        font=("Helvetica", int(CELL_PX * 0.24)),
                                        tags=tag)
            else:
                fsize = int(CELL_PX * 0.22)
                gap   = fsize + 2
                self.canvas.create_text(cx, cy - gap // 2,
                                        text=lines[0],
                                        fill="#a8e6cf",
                                        font=("Helvetica", fsize),
                                        tags=tag)
                self.canvas.create_text(cx, cy + gap // 2,
                                        text=lines[1],
                                        fill="#ffd3b6",
                                        font=("Helvetica", fsize),
                                        tags=tag)

    def _redraw_overlay(self):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self._draw_cell(r, c)

    # ── MOUSE INTERACTION ────────────────────────────────
    def _rc_from_event(self, event):
        c = event.x // CELL_PX
        r = event.y // CELL_PX
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return r, c
        return None, None

    def _place(self, r, c):
        mode = self.place_mode.get()
        if mode == "start":
            if self.start:
                sr, sc = self.start
                self.grid[sr][sc] = EMPTY
                self._draw_cell(sr, sc)
            self.start = (r, c)
            self.grid[r][c] = START
        elif mode == "goal":
            if self.goal:
                gr, gc = self.goal
                self.grid[gr][gc] = EMPTY
                self._draw_cell(gr, gc)
            self.goal = (r, c)
            self.grid[r][c] = GOAL
        else:
            if self.grid[r][c] == WALL:
                self.grid[r][c] = EMPTY
            elif self.grid[r][c] == EMPTY:
                self.grid[r][c] = WALL
        self._draw_cell(r, c)

    def _on_left_click(self, event):
        if self.running: return
        r, c = self._rc_from_event(event)
        if r is not None:
            self._place(r, c)

    def _on_left_drag(self, event):
        if self.running: return
        r, c = self._rc_from_event(event)
        if r is not None and self.place_mode.get() == "wall":
            if self.grid[r][c] == EMPTY:
                self.grid[r][c] = WALL
                self._draw_cell(r, c)

    def _on_right_click(self, event):
        if self.running: return
        r, c = self._rc_from_event(event)
        if r is not None and self.grid[r][c] == WALL:
            self.grid[r][c] = EMPTY
            self._draw_cell(r, c)

    # ── RANDOM MAZE ──────────────────────────────────────
    def _random_maze(self):
        self._stop()
        self.overlay    = {}
        self.cost_map   = {}
        self.anim_steps = []
        self.anim_index = 0
        self.last_algo  = None
        self._reset_results()

        N = GRID_SIZE

        g = [[WALL] * N for _ in range(N)]

        LOGICAL = (N - 1) // 2
        visited = [[False] * LOGICAL for _ in range(LOGICAL)]

        def carve(lr, lc):
            visited[lr][lc] = True
            gr, gc = lr * 2 + 1, lc * 2 + 1
            g[gr][gc] = EMPTY
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = lr + dr, lc + dc
                if 0 <= nr < LOGICAL and 0 <= nc < LOGICAL and not visited[nr][nc]:
                    wr, wc = gr + dr, gc + dc
                    g[wr][wc] = EMPTY
                    carve(nr, nc)

        carve(0, 0)

        num_shortcuts = random.randint(2, 3)
        shortcuts_added = 0
        attempts = 0
        while shortcuts_added < num_shortcuts and attempts < 200:
            attempts += 1
            r = random.randint(1, N - 2)
            c = random.randint(1, N - 2)
            if g[r][c] == WALL:
                if g[r][c - 1] == EMPTY and g[r][c + 1] == EMPTY:
                    g[r][c] = EMPTY
                    shortcuts_added += 1
                elif g[r - 1][c] == EMPTY and g[r + 1][c] == EMPTY:
                    g[r][c] = EMPTY
                    shortcuts_added += 1

        def pick_empty(r_lo, r_hi, c_lo, c_hi):
            cells = [(r, c) for r in range(r_lo, r_hi)
                             for c in range(c_lo, c_hi)
                             if g[r][c] == EMPTY]
            return random.choice(cells) if cells else None

        s  = pick_empty(1, N // 3 + 1,      1, N // 3 + 1)
        gl = pick_empty(N * 2 // 3, N - 1,  N * 2 // 3, N - 1)

        if s is None:
            for r in range(N):
                for c in range(N):
                    if g[r][c] == EMPTY:
                        s = (r, c); break
                if s: break
        if gl is None:
            for r in range(N - 1, -1, -1):
                for c in range(N - 1, -1, -1):
                    if g[r][c] == EMPTY and (r, c) != s:
                        gl = (r, c); break
                if gl: break

        g[s[0]][s[1]]   = START
        g[gl[0]][gl[1]] = GOAL

        self.grid  = g
        self.start = s
        self.goal  = gl

        self._draw_grid()
        self.status_var.set(
            f"Maze generated ({num_shortcuts} shortcuts carved). Click RUN to solve.")
        self.step_var.set("Step: —")

    # ── CONTROLS ─────────────────────────────────────────
    def _run(self):
        if not self.start or not self.goal:
            self.status_var.set("⚠  Place both Start (S) and Goal (G) first.")
            return

        self.running    = True
        self.paused     = False
        self.overlay    = {}
        self.cost_map   = {}
        self.anim_steps = []
        self.anim_index = 0
        self._reset_results()

        self.run_btn.config(state="disabled")
        self.pause_btn.config(state="normal", text="⏸  PAUSE", bg="#6c5ce7")
        self.stop_btn.config(state="normal")
        self.prev_btn.config(state="disabled")
        self.next_btn.config(state="disabled")

        algo = self.algo_var.get()
        if   algo == "A* Search":              algo_obj = AStarSearch(self.grid, self.start, self.goal)
        elif algo == "Hill Climbing":           algo_obj = HillClimbing(self.grid, self.start, self.goal)
        elif algo == "Simulated Annealing":     algo_obj = SimulatedAnnealing(self.grid, self.start, self.goal)
        elif algo == "Genetic Algorithm":       algo_obj = GeneticAlgorithm(self.grid, self.start, self.goal)
        elif algo == "Backtracking Search":     algo_obj = BacktrackingSearch(self.grid, self.start, self.goal)
        elif algo == "Arc Consistency (AC-3)":  algo_obj = ArcConsistency(self.grid, self.start, self.goal)

        self.anim_steps = algo_obj.solve()
        self.last_algo  = algo_obj

        self.anim_index = 0
        self._animate()

    def _animate(self):
        if not self.running or self.paused:
            return
        if self.anim_index >= len(self.anim_steps):
            self._finish()
            return
        self._apply_step(self.anim_index)
        self.anim_index += 1
        self.after_id = self.root.after(self.speed_var.get(), self._animate)

    def _apply_step(self, idx):
        overlay, cost_map, status = self.anim_steps[idx]
        self.overlay  = overlay
        self.cost_map = cost_map
        self._redraw_overlay()
        self.status_var.set(status)
        self.step_var.set(f"Step: {idx + 1} / {len(self.anim_steps)}")

    def _finish(self):
        self.running = False
        self.paused  = False
        self.run_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="⏸  PAUSE")
        self.stop_btn.config(state="disabled")
        self._update_nav_buttons()
        self._show_results()

    def _toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            if self.after_id:
                self.root.after_cancel(self.after_id)
                self.after_id = None
            self.pause_btn.config(text="▶  RESUME", bg="#00b894")
            self._update_nav_buttons()
            self.status_var.set("⏸  Paused. Use ◀ PREV / NEXT ▶ to navigate steps.")
        else:
            self.pause_btn.config(text="⏸  PAUSE", bg="#6c5ce7")
            self.prev_btn.config(state="disabled")
            self.next_btn.config(state="disabled")
            self._animate()

    def _stop(self):
        was_running = self.running
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.running = False
        self.paused  = False
        self.run_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="⏸  PAUSE", bg="#6c5ce7")
        self.stop_btn.config(state="disabled")
        self._update_nav_buttons()
        self.status_var.set("⏹  Stopped.")
        if was_running:
            self._show_results()

    def _step_back(self):
        if self.anim_index > 1:
            self.anim_index -= 1
            self._apply_step(self.anim_index - 1)
        self._update_nav_buttons()

    def _step_forward(self):
        if self.anim_index < len(self.anim_steps):
            self._apply_step(self.anim_index)
            self.anim_index += 1
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        can_nav = (not self.running or self.paused) and len(self.anim_steps) > 0
        self.prev_btn.config(state="normal" if (can_nav and self.anim_index > 1) else "disabled")
        self.next_btn.config(state="normal" if (can_nav and self.anim_index < len(self.anim_steps)) else "disabled")

    def _reset_results(self):
        self.cost_var.set("Total Cost Used: —")
        self.dist_var.set("Distance Traveled: —")
        self.optim_var.set("")

    def _show_results(self):
        algo = self.last_algo
        if algo is None:
            return
        if algo.total_cost is not None:
            label = "Total Cost Used" if algo.reached else "Cost So Far (not reached)"
            self.cost_var.set(f"{label}: {algo.total_cost} moves")
            self.dist_var.set(
                f"Distance Traveled: {algo.path_length} cells  "
                f"(straight-line minimum: {algo.straight_line})")
            if algo.reached:
                if algo.total_cost <= algo.straight_line:
                    self.optim_var.set("🏆 Optimal — matches the shortest possible distance.")
                else:
                    extra = algo.total_cost - algo.straight_line
                    self.optim_var.set(
                        f"↪ {extra} move(s) longer than the straight-line minimum "
                        f"(walls forced a detour).")
            else:
                self.optim_var.set("⚠ Did not reach the goal — stats reflect the partial path.")
        else:
            self.cost_var.set("Total Cost Used: —")
            self.dist_var.set(f"❌ No path found  (straight-line distance: {algo.straight_line})")
            self.optim_var.set("")

    def _reset(self):
        self._stop()
        self.grid       = [[EMPTY] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.start      = None
        self.goal       = None
        self.overlay    = {}
        self.cost_map   = {}
        self.anim_steps = []
        self.anim_index = 0
        self.last_algo  = None
        self._draw_grid()
        self.status_var.set("Grid cleared. Place Start, Goal, and Walls.")
        self.step_var.set("Step: —")
        self._reset_results()
        self.prev_btn.config(state="disabled")
        self.next_btn.config(state="disabled")


# ──────────────────────────────────────────────
#  ALGORITHM BASE
# ──────────────────────────────────────────────
class BaseAlgo:
    def __init__(self, grid, start, goal):
        self.grid  = [row[:] for row in grid]
        self.start = start
        self.goal  = goal
        self.steps = []

        self.reached       = False
        self.total_cost    = None
        self.path_length   = None
        self.straight_line = manhattan(start, goal)

    def snap(self, overlay, cost_map, status):
        self.steps.append((dict(overlay), dict(cost_map), status))

    def solve(self):
        raise NotImplementedError

    def _mark_path(self, path, overlay):
        for cell in path:
            if cell != self.start and cell != self.goal:
                overlay[cell] = C_PATH

    def _finalize(self, path, reached):
        self.reached = reached
        if path:
            self.path_length = len(path)
            self.total_cost  = max(len(path) - 1, 0)
        else:
            self.path_length = None
            self.total_cost  = None


# ──────────────────────────────────────────────
#  1. A* SEARCH
# ──────────────────────────────────────────────
class AStarSearch(BaseAlgo):
    def solve(self):
        g_score   = {self.start: 0}
        f_score   = {self.start: manhattan(self.start, self.goal)}
        came_from = {}
        closed    = set()

        counter   = 0
        open_heap = [(f_score[self.start], counter, self.start)]
        open_set  = {self.start}

        overlay  = {}
        cost_map = {}
        cost_map[self.start] = f"g=0\nf={f_score[self.start]}"

        self.snap(overlay, cost_map,
                  "A* Search started.  f = g (path cost) + h (Manhattan heuristic).")

        iteration = 0
        while open_heap:
            iteration += 1
            _, _, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            open_set.discard(current)
            closed.add(current)

            if current != self.start and current != self.goal:
                overlay[current] = C_CURRENT

            g_cur = g_score[current]
            h_cur = manhattan(current, self.goal)

            if current == self.goal:
                path  = []
                node  = self.goal
                while node is not None:
                    path.append(node)
                    node = came_from.get(node)
                path.reverse()
                self._mark_path(path, overlay)
                self._finalize(path, reached=True)
                self.snap(overlay, cost_map,
                          f"🎉 Goal reached!  Path length: {len(path)} cells  |  "
                          f"Total g-cost: {g_cur}  |  Iterations: {iteration}.")
                return self.steps

            self.snap(overlay, cost_map,
                      f"Iter {iteration}: Expand {current}  |  "
                      f"g={g_cur}, h={h_cur}, f={g_cur + h_cur}  |  "
                      f"Open: {len(open_set)}, Closed: {len(closed)}")

            if current != self.start:
                overlay[current] = C_VISITED

            for nb in neighbors(self.grid, *current):
                if nb in closed:
                    continue
                tentative_g = g_cur + 1
                if tentative_g < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb]   = tentative_g
                    h_nb          = manhattan(nb, self.goal)
                    f_nb          = tentative_g + h_nb
                    f_score[nb]   = f_nb
                    counter      += 1
                    heapq.heappush(open_heap, (f_nb, counter, nb))
                    open_set.add(nb)
                    cost_map[nb] = f"g={tentative_g}\nf={f_nb}"
                    if nb != self.goal:
                        overlay[nb] = C_FRONTIER

            if len(self.steps) > GRID_SIZE * GRID_SIZE * 4:
                break

        self._finalize(None, reached=False)
        self.snap(overlay, cost_map,
                  "No path found — goal is unreachable from start.")
        return self.steps


# ──────────────────────────────────────────────
#  2. HILL CLIMBING
# ──────────────────────────────────────────────
class HillClimbing(BaseAlgo):
    def solve(self):
        current  = self.start
        visited  = set()
        path     = [current]
        overlay  = {}
        cost_map = {}

        self.snap(overlay, cost_map,
                  "Hill Climbing started. Greedy move toward goal by Manhattan distance.")

        for _ in range(GRID_SIZE * GRID_SIZE * 2):
            visited.add(current)
            nbrs = neighbors(self.grid, *current)

            for v in visited:
                if v != self.start and v != self.goal:
                    overlay[v] = C_VISITED
            for n in nbrs:
                if n not in visited and n != self.goal:
                    overlay[n] = C_FRONTIER
            overlay[current] = C_CURRENT

            for n in nbrs:
                cost_map[n] = f"h={manhattan(n, self.goal)}"
            cost_map[current] = f"h={manhattan(current, self.goal)}"

            if current == self.goal:
                self._mark_path(path, overlay)
                self._finalize(path, reached=True)
                self.snap(overlay, cost_map, "🎉 Goal reached!")
                return self.steps

            nbrs.sort(key=lambda x: manhattan(x, self.goal))
            if not nbrs:
                self.snap(overlay, cost_map,
                          "⛔ No passable neighbours. Hill Climbing stuck.")
                break

            best_nbr  = nbrs[0]
            best_dist = manhattan(best_nbr, self.goal)
            cur_dist  = manhattan(current, self.goal)

            if best_dist >= cur_dist:
                overlay[current] = C_BACKTRACK
                self.snap(overlay, cost_map,
                          f"⚠ LOCAL MINIMUM at {current}!  "
                          f"All neighbours (h={best_dist}) ≥ current (h={cur_dist}). "
                          "Cannot escape without restart.")
                break

            self.snap(overlay, cost_map,
                      f"Move {current}→{best_nbr}  |  h: {cur_dist}→{best_dist}")
            current = best_nbr
            path.append(current)

        self._mark_path(path, overlay)
        self._finalize(path, reached=False)
        return self.steps


# ──────────────────────────────────────────────
#  3. SIMULATED ANNEALING
# ──────────────────────────────────────────────
class SimulatedAnnealing(BaseAlgo):
    def solve(self):
        current  = self.start
        visited  = {current}
        path     = [current]
        overlay  = {}
        cost_map = {}

        T     = 1.0
        T_min = 0.01
        alpha = 0.92
        step  = 0

        self.snap(overlay, cost_map,
                  f"Simulated Annealing started.  T={T:.3f}.")

        while current != self.goal and T > T_min:
            nbrs = neighbors(self.grid, *current)
            if not nbrs:
                break

            for v in visited:
                if v != self.start and v != self.goal:
                    overlay[v] = C_VISITED
            for n in nbrs:
                if n != self.goal:
                    overlay[n] = C_FRONTIER
                cost_map[n] = f"h={manhattan(n, self.goal)}"
            overlay[current]  = C_CURRENT
            cost_map[current] = f"h={manhattan(current, self.goal)}"

            candidate = random.choice(nbrs)
            delta = manhattan(current, self.goal) - manhattan(candidate, self.goal)

            if delta > 0:
                accepted = True
                msg = (f"T={T:.3f} | Better move {current}→{candidate}  "
                       f"Δh={delta:+d}. Accepted.")
            else:
                prob     = math.exp(delta / T) if T > 0 else 0
                accepted = random.random() < prob
                if accepted:
                    msg = (f"T={T:.3f} | ⚡ Worse move accepted! {current}→{candidate}  "
                           f"Δh={delta:+d}, p={prob:.2f}. Escaping local min!")
                else:
                    msg = (f"T={T:.3f} | Worse move rejected. {current}→{candidate}  "
                           f"Δh={delta:+d}, p={prob:.2f}.")

            self.snap(overlay, cost_map, msg)

            if accepted:
                current = candidate
                path.append(current)
                visited.add(current)

            T    *= alpha
            step += 1

        for v in visited:
            if v != self.start and v != self.goal:
                overlay[v] = C_VISITED
        overlay[current] = C_CURRENT

        if current == self.goal:
            self._mark_path(path, overlay)
            self._finalize(path, reached=True)
            self.snap(overlay, cost_map,
                      f"🎉 Goal reached after {step} steps!  T={T:.4f}.")
        else:
            self._mark_path(path, overlay)
            self._finalize(path, reached=False)
            self.snap(overlay, cost_map,
                      f"Temperature cooled to {T:.4f}.  Ended at {current}.")
        return self.steps


# ──────────────────────────────────────────────
#  4. GENETIC ALGORITHM
# ──────────────────────────────────────────────
class GeneticAlgorithm(BaseAlgo):
    POP_SIZE    = 20
    PATH_LEN    = GRID_SIZE * 2
    GENERATIONS = 40
    MUTATION_R  = 0.15

    def _random_path(self):
        return [random.randint(0, 3) for _ in range(self.PATH_LEN)]

    def _follow(self, genome):
        r, c  = self.start
        cells = [(r, c)]
        for gene in genome:
            if (r, c) == self.goal:
                break
            dr, dc = DIRECTIONS[gene]
            nr, nc = r + dr, c + dc
            if is_passable(self.grid, nr, nc):
                r, c = nr, nc
            cells.append((r, c))
        return (r, c), cells

    def _fitness(self, genome):
        (r, c), _ = self._follow(genome)
        dist  = manhattan((r, c), self.goal)
        score = GRID_SIZE * 3 - dist
        if (r, c) == self.goal:
            score += 100
        return score

    def solve(self):
        pop = [self._random_path() for _ in range(self.POP_SIZE)]

        for gen in range(self.GENERATIONS):
            scored = sorted(pop, key=self._fitness, reverse=True)
            best   = scored[0]
            _, best_cells = self._follow(best)
            bf     = self._fitness(best)

            overlay  = {}
            cost_map = {}
            sample   = random.sample(scored, min(5, len(scored)))
            for ind in sample:
                _, cells = self._follow(ind)
                for cell in cells:
                    if cell != self.start and cell != self.goal:
                        overlay[cell] = C_GA_POP
            for cell in best_cells:
                if cell != self.start and cell != self.goal:
                    overlay[cell] = C_GA_BEST

            best_end, _ = self._follow(best)
            cost_map[best_end] = f"fit={bf}"

            if best_end == self.goal:
                status = f"🎉 Gen {gen + 1}: GOAL REACHED!  Fitness={bf}"
            else:
                status = (f"Gen {gen + 1}/{self.GENERATIONS}  |  "
                          f"Best fitness={bf}  |  "
                          f"Best end={best_end}  |  "
                          f"dist={manhattan(best_end, self.goal)}")
            self.snap(overlay, cost_map, status)

            if best_end == self.goal:
                break

            parents  = scored[:self.POP_SIZE // 2]
            children = list(parents)
            while len(children) < self.POP_SIZE:
                p1, p2 = random.sample(parents, 2)
                pt     = random.randint(1, self.PATH_LEN - 1)
                child  = p1[:pt] + p2[pt:]
                for i in range(len(child)):
                    if random.random() < self.MUTATION_R:
                        child[i] = random.randint(0, 3)
                children.append(child)
            pop = children

        scored = sorted(pop, key=self._fitness, reverse=True)
        best_end, best_cells = self._follow(scored[0])
        overlay = {}
        for cell in best_cells:
            if cell != self.start and cell != self.goal:
                overlay[cell] = C_PATH
        dist = manhattan(best_end, self.goal)
        self._finalize(best_cells, reached=(best_end == self.goal))
        self.snap(overlay, {}, f"GA complete.  Final best path shown.  Dist to goal: {dist}.")
        return self.steps


# ──────────────────────────────────────────────
#  5. BACKTRACKING SEARCH (DFS)
# ──────────────────────────────────────────────
class BacktrackingSearch(BaseAlgo):
    def solve(self):
        visited = set()
        path    = []
        found   = [False]

        def dfs(cell):
            if found[0]:
                return
            r, c = cell
            if not in_bounds(r, c) or self.grid[r][c] == WALL or cell in visited:
                return

            visited.add(cell)
            path.append(cell)

            ov = {}
            cm = {}
            for v in visited:
                if v != self.start and v != self.goal:
                    ov[v] = C_VISITED
            for p in path:
                if p != self.start and p != self.goal:
                    ov[p] = C_FRONTIER
            ov[cell]  = C_CURRENT
            cm[cell]  = f"d={len(path)}"

            if cell == self.goal:
                found[0] = True
                self._mark_path(path, ov)
                self._finalize(path, reached=True)
                self.steps.append((dict(ov), dict(cm),
                                   f"🎉 GOAL FOUND at {cell}!  Path length: {len(path)}."))
                return

            self.steps.append((dict(ov), dict(cm),
                               f"Exploring {cell}.  Path depth: {len(path)}."))

            nbrs = neighbors(self.grid, r, c)
            random.shuffle(nbrs)
            for nbr in nbrs:
                if nbr not in visited:
                    dfs(nbr)
                    if found[0]:
                        return

            if not found[0]:
                path.pop()
                ov2 = {}
                cm2 = {}
                for v in visited:
                    if v != self.start and v != self.goal:
                        ov2[v] = C_VISITED
                for p in path:
                    if p != self.start and p != self.goal:
                        ov2[p] = C_FRONTIER
                if path:
                    ov2[path[-1]] = C_CURRENT
                    cm2[path[-1]] = f"d={len(path)}"
                ov2[cell] = C_BACKTRACK
                self.steps.append((dict(ov2), dict(cm2),
                                   f"⬅  BACKTRACKING from {cell}  →  "
                                   f"{path[-1] if path else self.start}"))

        random.seed(42)
        dfs(self.start)
        if not found[0]:
            self._finalize(None, reached=False)
            self.steps.append(({}, {}, "No path found between Start and Goal."))
        return self.steps


# ──────────────────────────────────────────────
#  6. ARC CONSISTENCY (AC-3 inspired)
# ──────────────────────────────────────────────
class ArcConsistency(BaseAlgo):
    def solve(self):
        # Phase 1 — backward BFS from Goal
        reachable = set()
        queue     = collections.deque([self.goal])
        reachable.add(self.goal)
        while queue:
            cell = queue.popleft()
            for nb in neighbors(self.grid, *cell):
                if nb not in reachable:
                    reachable.add(nb)
                    queue.append(nb)

        pruned = set()
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.grid[r][c] != WALL and (r, c) not in reachable:
                    pruned.add((r, c))

        ov_pruned   = {cell: C_PRUNED  for cell in pruned}
        cost_pruned = {cell: "pruned"  for cell in pruned}

        self.snap(ov_pruned, cost_pruned,
                  f"AC-3 Phase 1: {len(pruned)} cells pruned.  "
                  f"{len(reachable)} cells remain in consistent domain.")

        # Phase 2 — DFS on consistent domain
        visited = set()
        path    = []
        found   = [False]

        def dfs(cell):
            if found[0]:
                return
            if cell in pruned or cell in visited:
                return
            r, c = cell
            if not in_bounds(r, c) or self.grid[r][c] == WALL:
                return

            visited.add(cell)
            path.append(cell)

            ov = dict(ov_pruned)
            cm = dict(cost_pruned)
            for v in visited:
                if v != self.start and v != self.goal and v not in pruned:
                    ov[v] = C_VISITED
            for p in path:
                if p != self.start and p != self.goal:
                    ov[p] = C_FRONTIER
            ov[cell]  = C_CURRENT
            cm[cell]  = f"d={len(path)}"

            if cell == self.goal:
                found[0] = True
                self._mark_path(path, ov)
                self._finalize(path, reached=True)
                self.steps.append((dict(ov), dict(cm),
                                   f"🎉 GOAL FOUND!  AC-3 path length: {len(path)}.  "
                                   "Pruning eliminated bad branches automatically."))
                return

            self.steps.append((dict(ov), dict(cm),
                               f"AC-3 DFS at {cell}  (depth={len(path)}).  "
                               "Pruned cells auto-skipped."))

            for nb in neighbors(self.grid, r, c):
                if nb not in pruned and nb not in visited:
                    dfs(nb)
                    if found[0]:
                        return

            if not found[0]:
                path.pop()
                ov2 = dict(ov_pruned)
                cm2 = dict(cost_pruned)
                for v in visited:
                    if v != self.start and v != self.goal and v not in pruned:
                        ov2[v] = C_VISITED
                for p in path:
                    if p != self.start and p != self.goal:
                        ov2[p] = C_FRONTIER
                if path:
                    ov2[path[-1]] = C_CURRENT
                ov2[cell] = C_BACKTRACK
                self.steps.append((dict(ov2), dict(cm2),
                                   f"⬅  AC-3 backtrack from {cell}."))

        dfs(self.start)
        if not found[0]:
            self._finalize(None, reached=False)
            self.steps.append((dict(ov_pruned), dict(cost_pruned),
                               "No path found (start/goal may be isolated by walls)."))
        return self.steps


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TCombobox",
                    fieldbackground=C_ACCENT,
                    background=C_ACCENT,
                    foreground=C_TEXT_MAIN,
                    selectbackground=C_ACCENT,
                    selectforeground=C_TEXT_MAIN)
    style.configure("TSeparator", background=C_ACCENT)

    app = PathfindingApp(root)
    root.mainloop()
