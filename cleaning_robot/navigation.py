"""Small obstacle-aware grid router for the bounded simulation arena.

This is not a SLAM or Nav2 replacement. Obstacles are fixed-frame, persistent
observations; every route is checked with the requested collision clearance.
"""
import heapq
import math
from typing import Optional

import numpy as np


class ArenaNavigator:
    def __init__(self, width_m: float, height_m: float, resolution_m: float = 0.04,
                 margin_m: float = 0.015):
        self.width = width_m
        self.height = height_m
        self.resolution = resolution_m
        self.margin = margin_m
        if min(width_m, height_m, resolution_m) <= 0 or margin_m < 0:
            raise ValueError("invalid arena navigation dimensions")
        self.obstacles = {}
        self.revision = 0

    def observe(self, detections, removed_ids=()):
        removed = set(removed_ids)
        changed = False
        for track_id in removed:
            changed |= self.obstacles.pop(track_id, None) is not None
        for det in detections:
            if det.kind != "debris" or det.track_id is None or det.track_id in removed:
                continue
            obstacle = (np.array(det.position_robot[:2], copy=True), det.radius_m)
            previous = self.obstacles.get(det.track_id)
            if (previous is None or np.linalg.norm(previous[0] - obstacle[0]) > 0.02
                    or abs(previous[1] - obstacle[1]) > 0.01):
                self.obstacles[det.track_id] = obstacle
                changed = True
        if changed:
            self.revision += 1

    def point_clear(self, point, radius: float, ignore_id=None) -> bool:
        x, y = point
        clearance = radius + self.margin
        if not (clearance <= x <= self.width - clearance
                and clearance <= y <= self.height - clearance):
            return False
        return all(key == ignore_id or math.hypot(x - centre[0], y - centre[1])
                   >= clearance + obstacle_radius
                   for key, (centre, obstacle_radius) in self.obstacles.items())

    def segment_clear(self, start, goal, radius: float, ignore_id=None) -> bool:
        if not self.point_clear(start, radius, ignore_id) or not self.point_clear(goal, radius, ignore_id):
            return False
        # The eroded arena is convex. Endpoint bounds and exact segment-circle
        # distances therefore check the entire segment without sampling holes.
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        length_sq = dx * dx + dy * dy
        for key, (centre, obstacle_radius) in self.obstacles.items():
            if key == ignore_id:
                continue
            t = ((centre[0] - start[0]) * dx + (centre[1] - start[1]) * dy) / max(length_sq, 1e-12)
            t = min(1.0, max(0.0, t))
            distance = math.hypot(start[0] + t * dx - centre[0], start[1] + t * dy - centre[1])
            if distance < radius + self.margin + obstacle_radius:
                return False
        return True

    def route(self, start, goal, radius: float, ignore_id=None) -> Optional[np.ndarray]:
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        if not self.point_clear(start, radius, ignore_id) or not self.point_clear(goal, radius, ignore_id):
            return None
        if self.segment_clear(start, goal, radius, ignore_id):
            return np.array([start, goal])

        scale = self.resolution
        def point(node):
            return np.array(node, dtype=float) * scale

        def nearby(value):
            centre = np.rint(value / scale).astype(int)
            options = [(int(centre[0] + dx), int(centre[1] + dy))
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
            return sorted((n for n in options
                           if self.segment_clear(value, point(n), radius, ignore_id)),
                          key=lambda n: np.linalg.norm(point(n) - value))

        starts, goals = nearby(start), nearby(goal)
        if not starts or not goals:
            return None
        goal_nodes = set(goals)
        costs, parents, queue = {}, {}, []
        for node in starts:
            costs[node] = float(np.linalg.norm(point(node) - start))
            heapq.heappush(queue, (costs[node] + np.linalg.norm(point(node) - goal), node))
        visited = set()
        final = None
        while queue:
            _, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            if node in goal_nodes:
                final = node
                break
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nxt = (node[0] + dx, node[1] + dy)
                    if nxt in visited or not self.segment_clear(point(node), point(nxt), radius, ignore_id):
                        continue
                    cost = costs[node] + scale * math.hypot(dx, dy)
                    if cost < costs.get(nxt, float("inf")):
                        costs[nxt], parents[nxt] = cost, node
                        estimate = cost + float(np.linalg.norm(point(nxt) - goal))
                        heapq.heappush(queue, (estimate, nxt))
        if final is None:
            return None
        nodes = [final]
        while nodes[-1] in parents:
            nodes.append(parents[nodes[-1]])
        raw = [start] + [point(n) for n in reversed(nodes)] + [goal]
        # Greedy visibility simplification keeps every resulting segment clear.
        result, index = [raw[0]], 0
        while index < len(raw) - 1:
            nxt = len(raw) - 1
            while nxt > index + 1 and not self.segment_clear(raw[index], raw[nxt], radius, ignore_id):
                nxt -= 1
            result.append(raw[nxt])
            index = nxt
        return np.asarray(result)
