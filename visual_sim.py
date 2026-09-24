"""
visual_sim.py  —  Autonomous Roomba Mop Visual Simulator v3
===========================================================
Self-contained 2D simulation with a clean, flicker-free state machine.
No external MissionPlanner dependency — this is a standalone demo that
proves the coverage + divert + scrub + resume architecture works.

Controls:
  [SPACE]  Pause / Resume
  [S]      Spawn random stain
  [R]      Reset
  [Q/ESC]  Quit
"""

import cv2
import numpy as np
from enum import Enum, auto
from typing import List, Optional, Tuple


# ── Configuration ────────────────────────────────────────────────────────────

ARENA_W = 2.4          # meters
ARENA_H = 1.7
SCALE   = 280          # px per meter
HUD_W   = 350

ROBOT_R       = 0.17   # mop radius (meters)
WALL_MARGIN   = ROBOT_R + 0.06
MAX_V         = 0.30   # coverage cruise m/s
NAV_V         = 0.25   # divert speed
SCRUB_V       = 0.18   # scrub speed
MAX_W         = 2.0    # rad/s

LANE_OVERLAP  = 0.15
LOOKAHEAD     = 0.18
GOAL_TOL      = 0.05
SCRUB_DETECT  = 1.0    # camera detection range (m)
SCRUB_FOV_DEG = 35     # half-angle of camera cone

DT = 0.033             # ~30 FPS sim


# ── State Machine ────────────────────────────────────────────────────────────

class State(Enum):
    COVERAGE  = auto()
    DIVERTING = auto()
    SCRUBBING = auto()
    DONE      = auto()


# ── Path Generators ──────────────────────────────────────────────────────────

def make_coverage_path(x0, x1, y0, y1, robot_r, overlap, step=0.03):
    """Dense boustrophedon (lawnmower) waypoints."""
    lane_sp = 2 * robot_r * (1 - overlap)
    ys = np.arange(y0, y1 + 1e-9, lane_sp)
    pts = []
    for i, y in enumerate(ys):
        if i % 2 == 0:
            xs = np.arange(x0, x1 + 1e-9, step)
        else:
            xs = np.arange(x1, x0 - 1e-9, -step)
        for x in xs:
            pts.append((x, y))
    return np.array(pts)


def make_spiral(robot_r, stain_r, margin=0.02, pts_per_rev=20):
    """Outward Archimedean spiral centered at (0,0)."""
    eff_r = stain_r + margin
    if eff_r <= robot_r:
        # One footprint covers the stain — small agitation circle
        th = np.linspace(0, 2 * np.pi, pts_per_rev)
        r = 0.25 * robot_r
        return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    pitch = robot_r * 0.65          # tighter spiral = faster coverage
    b = pitch / (2 * np.pi)
    n_turns = max(int(np.ceil(eff_r / pitch)), 1)
    theta_max = n_turns * 2 * np.pi
    thetas = np.linspace(0, theta_max, n_turns * pts_per_rev)
    r = b * thetas
    return np.stack([r * np.cos(thetas), r * np.sin(thetas)], axis=1)


# ── Pure Pursuit (robust) ───────────────────────────────────────────────────

def steer_to_path(pose, path, progress, speed, lookahead=LOOKAHEAD):
    """Returns (v, w, new_progress, finished)."""
    x, y, th = pose
    pos = np.array([x, y])
    n = len(path)
    if n == 0:
        return 0., 0., 0, True

    # Advance progress along segments
    while progress < n - 2:
        p1, p2 = path[progress], path[progress + 1]
        seg = p2 - p1
        seg2 = np.dot(seg, seg)
        if seg2 < 1e-8:
            progress += 1; continue
        t = np.dot(pos - p1, seg) / seg2
        if t >= 0.92 or np.linalg.norm(pos - p2) < lookahead * 0.35:
            progress += 1
        else:
            break

    # Goal check
    d_goal = float(np.linalg.norm(path[-1] - pos))
    if progress >= n - 2 and d_goal < GOAL_TOL:
        return 0., 0., progress, True

    # Lookahead along polyline
    idx = min(progress, n - 2)
    p1, p2 = path[idx], path[idx + 1]
    seg = p2 - p1
    seg_len = float(np.linalg.norm(seg))
    t = float(np.clip(np.dot(pos - p1, seg) / max(seg_len**2, 1e-8), 0, 1))
    remain = (1 - t) * seg_len
    if remain >= lookahead:
        target = p1 + (t + lookahead / max(seg_len, 1e-6)) * seg
    else:
        need = lookahead - remain
        target = path[-1]
        for k in range(idx + 1, n - 1):
            sl = float(np.linalg.norm(path[k+1] - path[k]))
            if sl >= need:
                target = path[k] + (path[k+1] - path[k]) * (need / sl)
                break
            need -= sl

    # Steering
    dx, dy = target[0] - x, target[1] - y
    tgt_ang = np.arctan2(dy, dx)
    err = (tgt_ang - th + np.pi) % (2 * np.pi) - np.pi

    if abs(err) > np.deg2rad(50):
        v = 0.0
        w = float(np.clip(2.5 * err, -MAX_W, MAX_W))
        if abs(w) < 0.5:
            w = float(np.sign(err) * 0.6)
    else:
        curv = 2.0 * np.sin(err) / max(lookahead, 0.04)
        v = speed * max(0.15, np.cos(err))
        if d_goal < 0.25 and progress >= n - 2:
            v = min(v, max(0.06, d_goal * 2))
        w = float(np.clip(v * curv + 0.8 * err, -MAX_W, MAX_W))

    return float(v), float(w), progress, False


# ── Simulator ────────────────────────────────────────────────────────────────

class Sim:
    def __init__(self):
        self.W = int(ARENA_W * SCALE)
        self.H = int(ARENA_H * SCALE)

        # Coverage path
        self.cov_path = make_coverage_path(
            WALL_MARGIN, ARENA_W - WALL_MARGIN,
            WALL_MARGIN, ARENA_H - WALL_MARGIN,
            ROBOT_R, LANE_OVERLAP)
        self.cov_idx = 0

        # State
        self.state = State.COVERAGE
        self.pose = (WALL_MARGIN, WALL_MARGIN, 0.0)
        self.sim_t = 0.0
        self.paused = False

        # Scrub
        self.scrub_path = None
        self.scrub_idx = 0
        self.scrub_target_id = None
        self.resume_cov_idx = 0

        # Stains  {id, pos, radius, cleaned}
        self.stains = [
            {"id": 1, "pos": np.array([1.05, 0.35]), "r": 0.09, "r0": 0.09, "cleaned": False},
            {"id": 2, "pos": np.array([1.70, 0.80]), "r": 0.08, "r0": 0.08, "cleaned": False},
            {"id": 3, "pos": np.array([0.80, 1.20]), "r": 0.10, "r0": 0.10, "cleaned": False},
        ]
        self.next_id = 4
        self.handled_ids = set()

        # Debris
        self.debris_pos = np.array([1.40, 1.10])
        self.debris_r = 0.04

        # Trail
        self.trail = np.zeros((self.H, self.W), dtype=np.uint8)

    # ── helpers ──

    def w2px(self, xm, ym):
        return (int(np.clip(xm * SCALE, 0, self.W-1)),
                int(np.clip(ym * SCALE, 0, self.H-1)))

    def _visible_stains(self):
        """Return stains inside the forward camera cone."""
        rx, ry, rth = self.pose
        c, s = np.cos(rth), np.sin(rth)
        R = np.array([[c, -s], [s, c]])
        vis = []
        for st in self.stains:
            if st["cleaned"] or st["id"] in self.handled_ids:
                continue
            rel = R.T @ (st["pos"] - np.array([rx, ry]))
            if 0.12 <= rel[0] <= SCRUB_DETECT and abs(rel[1]) <= rel[0] * np.tan(np.deg2rad(SCRUB_FOV_DEG)):
                vis.append(st)
        return vis

    def _closest_visible_stain(self):
        vis = self._visible_stains()
        if not vis:
            return None
        pos = np.array(self.pose[:2])
        dists = [np.linalg.norm(s["pos"] - pos) for s in vis]
        return vis[int(np.argmin(dists))]

    # ── step ──

    def step(self):
        if self.paused:
            return 0., 0.

        x, y, th = self.pose
        v, w = 0., 0.

        if self.state == State.COVERAGE:
            # Check for nearby stain to divert to
            tgt = self._closest_visible_stain()
            if tgt is not None and np.linalg.norm(tgt["pos"] - np.array([x, y])) < 1.2:
                self.scrub_target_id = tgt["id"]
                self.resume_cov_idx = self.cov_idx
                self.state = State.DIVERTING
                return self.step()

            v, w, self.cov_idx, done = steer_to_path(
                self.pose, self.cov_path, self.cov_idx, MAX_V)
            if done:
                self.state = State.DONE
                v, w = 0., 0.

        elif self.state == State.DIVERTING:
            tgt_st = next((s for s in self.stains if s["id"] == self.scrub_target_id), None)
            if tgt_st is None or tgt_st["cleaned"]:
                self.state = State.COVERAGE
                return self.step()
            goal = tgt_st["pos"]
            dist = float(np.linalg.norm(goal - np.array([x, y])))
            if dist < ROBOT_R * 0.6:
                # Arrived — start scrub
                local = make_spiral(ROBOT_R, tgt_st["r"])
                self.scrub_path = local + goal
                self.scrub_idx = 0
                self.state = State.SCRUBBING
                return self.step()
            # Steer toward stain
            dx, dy = goal[0] - x, goal[1] - y
            tgt_ang = np.arctan2(dy, dx)
            err = (tgt_ang - th + np.pi) % (2 * np.pi) - np.pi
            if abs(err) > np.deg2rad(45):
                v, w = 0., float(np.clip(2.5 * err, -MAX_W, MAX_W))
            else:
                v = NAV_V * max(0.2, np.cos(err))
                w = float(np.clip(2.0 * err, -MAX_W, MAX_W))

        elif self.state == State.SCRUBBING:
            v, w, self.scrub_idx, done = steer_to_path(
                self.pose, self.scrub_path, self.scrub_idx, SCRUB_V, lookahead=0.12)
            if done:
                self.handled_ids.add(self.scrub_target_id)
                self.scrub_target_id = None
                self.scrub_path = None
                # Resume coverage at nearest forward waypoint
                pos = np.array([x, y])
                search = self.cov_path[self.resume_cov_idx:min(len(self.cov_path), self.resume_cov_idx + 80)]
                if len(search) > 0:
                    d = np.linalg.norm(search - pos, axis=1)
                    self.cov_idx = self.resume_cov_idx + int(np.argmin(d)) + 2
                    self.cov_idx = min(self.cov_idx, len(self.cov_path) - 1)
                else:
                    self.cov_idx = self.resume_cov_idx
                self.state = State.COVERAGE
                return 0., 0.

        # ── Integrate kinematics ──
        th += w * DT
        x += v * np.cos(th) * DT
        y += v * np.sin(th) * DT
        x = float(np.clip(x, ROBOT_R + 0.01, ARENA_W - ROBOT_R - 0.01))
        y = float(np.clip(y, ROBOT_R + 0.01, ARENA_H - ROBOT_R - 0.01))
        self.pose = (x, y, th)
        self.sim_t += DT

        # ── Mop trail ──
        px, py = self.w2px(x, y)
        cv2.circle(self.trail, (px, py), int(ROBOT_R * SCALE), 255, -1)

        # ── Stain erosion ──
        for st in self.stains:
            if st["cleaned"]:
                continue
            d = np.linalg.norm(np.array([x, y]) - st["pos"])
            if d < ROBOT_R + 0.04:
                rate = 0.15 if self.state == State.SCRUBBING else 0.05
                st["r"] -= rate * DT
                if st["r"] <= 0.008:
                    st["cleaned"] = True
                    st["r"] = 0.0

        return v, w

    # ── render ──

    def render(self, v, w):
        # Floor
        arena = np.full((self.H, self.W, 3), 235, dtype=np.uint8)
        gs = int(0.5 * SCALE)
        for gx in range(0, self.W, gs):
            cv2.line(arena, (gx, 0), (gx, self.H), (222, 222, 222), 1)
        for gy in range(0, self.H, gs):
            cv2.line(arena, (0, gy), (self.W, gy), (222, 222, 222), 1)

        # Clean trail
        m = self.trail > 0
        arena[m] = (arena[m] * 0.72 + np.array([252, 248, 230]) * 0.28).astype(np.uint8)

        # Walls
        cv2.rectangle(arena, (0, 0), (self.W - 1, self.H - 1), (75, 75, 80), 6)

        # Coverage path (thin)
        if len(self.cov_path) > 1:
            for i in range(0, len(self.cov_path) - 1, 5):
                j = min(i + 5, len(self.cov_path) - 1)
                cv2.line(arena, self.w2px(*self.cov_path[i]), self.w2px(*self.cov_path[j]), (210, 210, 210), 1, cv2.LINE_AA)

        # Active scrub spiral
        if self.state == State.SCRUBBING and self.scrub_path is not None:
            for i in range(len(self.scrub_path) - 1):
                cv2.line(arena, self.w2px(*self.scrub_path[i]), self.w2px(*self.scrub_path[i+1]),
                         (0, 130, 255), 2, cv2.LINE_AA)
        elif self.state == State.DIVERTING and self.scrub_target_id is not None:
            tgt_st = next((s for s in self.stains if s["id"] == self.scrub_target_id), None)
            if tgt_st:
                cv2.line(arena, self.w2px(*self.pose[:2]), self.w2px(*tgt_st["pos"]),
                         (0, 200, 255), 2, cv2.LINE_AA)

        # Stains
        for st in self.stains:
            if not st["cleaned"]:
                sp = self.w2px(*st["pos"])
                rp = max(3, int(st["r"] * SCALE))
                cv2.circle(arena, sp, rp, (25, 55, 155), -1, cv2.LINE_AA)
                cv2.circle(arena, sp, rp + 2, (10, 30, 100), 1, cv2.LINE_AA)
                cv2.putText(arena, "Stain", (sp[0] - 18, sp[1] - rp - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (15, 35, 130), 1)

        # Debris
        dp = self.w2px(*self.debris_pos)
        dr = int(self.debris_r * SCALE)
        cv2.circle(arena, dp, dr, (40, 180, 240), -1, cv2.LINE_AA)
        cv2.circle(arena, dp, dr, (10, 100, 180), 2, cv2.LINE_AA)
        cv2.putText(arena, "3D Debris", (dp[0] - 28, dp[1] - dr - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 90, 170), 1)

        # Camera FOV cone
        rx, ry, rth = self.pose
        rp = self.w2px(rx, ry)
        fd = SCRUB_DETECT * SCALE
        hf = np.deg2rad(SCRUB_FOV_DEG)
        fp1 = (int(rp[0] + fd * np.cos(rth - hf)), int(rp[1] + fd * np.sin(rth - hf)))
        fp2 = (int(rp[0] + fd * np.cos(rth + hf)), int(rp[1] + fd * np.sin(rth + hf)))
        overlay = arena.copy()
        cv2.fillPoly(overlay, [np.array([rp, fp1, fp2], dtype=np.int32)], (255, 232, 205))
        cv2.addWeighted(overlay, 0.30, arena, 0.70, 0, arena)
        cv2.polylines(arena, [np.array([rp, fp1, fp2], dtype=np.int32)], True, (235, 185, 125), 1, cv2.LINE_AA)

        # Robot chassis
        mr = int(ROBOT_R * SCALE)
        cv2.circle(arena, rp, mr, (55, 155, 55), 3, cv2.LINE_AA)
        cv2.circle(arena, rp, int(mr * 0.85), (48, 48, 52), -1, cv2.LINE_AA)
        cv2.circle(arena, rp, int(mr * 0.38), (75, 75, 80), -1, cv2.LINE_AA)
        cam = (int(rp[0] + 0.08 * SCALE * np.cos(rth)), int(rp[1] + 0.08 * SCALE * np.sin(rth)))
        cv2.circle(arena, cam, 4, (0, 215, 255), -1, cv2.LINE_AA)
        hp = (int(rp[0] + (mr + 14) * np.cos(rth)), int(rp[1] + (mr + 14) * np.sin(rth)))
        cv2.arrowedLine(arena, rp, hp, (0, 255, 255), 2, tipLength=0.30)

        # ── HUD ──
        hud = np.full((self.H, HUD_W, 3), (26, 26, 30), dtype=np.uint8)

        # Header
        cv2.rectangle(hud, (0, 0), (HUD_W, 54), (38, 38, 44), -1)
        cv2.putText(hud, "ROOMBA MOP ROBOT", (16, 27), cv2.FONT_HERSHEY_DUPLEX, 0.54, (255, 255, 255), 1)
        cv2.putText(hud, "RGB-D Autonomous Perception HUD", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (155, 155, 165), 1)

        # State box
        st = self.state
        sc = {State.COVERAGE: (0, 230, 100), State.DIVERTING: (0, 165, 255),
              State.SCRUBBING: (0, 100, 255), State.DONE: (180, 180, 180)}[st]
        cv2.rectangle(hud, (14, 66), (HUD_W - 14, 112), (35, 35, 42), -1)
        cv2.rectangle(hud, (14, 66), (HUD_W - 14, 112), sc, 1)
        cv2.putText(hud, "MISSION STATE:", (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (175, 175, 175), 1)
        cv2.putText(hud, st.name, (24, 104), cv2.FONT_HERSHEY_DUPLEX, 0.58, sc, 2)

        # Metrics
        wd = min(rx - ROBOT_R, ARENA_W - rx - ROBOT_R, ry - ROBOT_R, ARENA_H - ry - ROBOT_R)
        wc = (0, 255, 100) if wd > 0.10 else (0, 165, 255) if wd > 0.04 else (0, 0, 255)
        cov = min(100., np.count_nonzero(self.trail) / (self.W * self.H) * 100 * 1.45)
        rows = [
            ("Linear Vel (v)",   f"{v:.2f} m/s",            (235, 235, 235)),
            ("Angular (w)",      f"{w:+.2f} rad/s",         (235, 235, 235)),
            ("Wall Clearance",   f"{wd*100:.1f} cm",        wc),
            ("Pose X, Y",       f"({rx:.2f}, {ry:.2f}) m",  (195, 215, 255)),
            ("Sim Time",         f"{self.sim_t:.1f} s",     (195, 195, 195)),
            ("Floor Coverage",   f"{cov:.1f} %",            (95, 255, 175)),
        ]
        y = 140
        for lbl, val, c in rows:
            cv2.putText(hud, lbl, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (145, 145, 155), 1)
            cv2.putText(hud, val, (HUD_W - 135, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, c, 1)
            y += 23

        # Stain status
        y += 8
        cv2.rectangle(hud, (14, y), (HUD_W - 14, y + 18 + 16 * len(self.stains)), (35, 35, 42), -1)
        cv2.putText(hud, "TARGET STATUS:", (24, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (175, 175, 175), 1)
        for i, st_data in enumerate(self.stains):
            txt = "CLEANED" if st_data["cleaned"] else f"Radius: {st_data['r']*100:.1f}cm"
            col = (0, 255, 100) if st_data["cleaned"] else (50, 100, 255)
            cv2.putText(hud, f"Stain #{i+1}: {txt}", (24, y + 32 + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)

        # Controls
        y += 40 + 15 * len(self.stains)
        cv2.putText(hud, "CONTROLS:", (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 255, 255), 1)
        y += 18
        for k, d in [("[SPACE]", "Pause / Resume"), ("[S]", "Spawn Stain"),
                      ("[R]", "Reset"), ("[Q]", "Quit")]:
            cv2.putText(hud, k, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 215, 255), 1)
            cv2.putText(hud, d, (92, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (165, 165, 175), 1)
            y += 17

        return np.hstack([arena, hud])

    def spawn_stain(self):
        self.stains.append({
            "id": self.next_id,
            "pos": np.array([np.random.uniform(0.45, ARENA_W - 0.45),
                             np.random.uniform(0.45, ARENA_H - 0.45)]),
            "r": 0.09, "r0": 0.09, "cleaned": False,
        })
        self.next_id += 1


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    sim = Sim()
    win = "Autonomous Roomba Mop  -  Visual Simulator v3"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("=" * 55)
    print("  Roomba Mop Robot  -  Visual Simulator v3")
    print("  [SPACE] Pause   [S] Stain   [R] Reset   [Q] Quit")
    print("=" * 55)

    while True:
        v, w = sim.step()
        frame = sim.render(v, w)
        cv2.imshow(win, frame)
        key = cv2.waitKey(int(DT * 1000)) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            sim.paused = not sim.paused
        elif key == ord('s'):
            sim.spawn_stain()
        elif key == ord('r'):
            sim = Sim()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
