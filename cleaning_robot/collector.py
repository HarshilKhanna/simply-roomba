"""Collector command/feedback contract shared by the planner and simulation.

There is deliberately no motor driver here. A hardware adapter must report
completion and payload evidence; elapsed time alone never proves a pickup.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from .config import CollectorConfig
from .perception import Detection


LITTER_CLASSES = ("plastic_bottle", "wrapper", "paper_cup", "paper_ball")


class CollectorCommand(Enum):
    STOW = auto()
    LOWER = auto()
    CLOSE = auto()
    LIFT = auto()
    HOLD = auto()


@dataclass(frozen=True)
class CollectorFeedback:
    command: CollectorCommand = CollectorCommand.STOW
    completed: bool = False
    payload_present: Optional[bool] = None
    fault: Optional[str] = None


def pickup_rejection(det: Detection, cfg: CollectorConfig) -> Optional[str]:
    """Return why a target is unsuitable, or None for a known lightweight item.

    The simulation supplies labelled dimensions and masses. An RGB-D camera
    cannot measure mass; real operation needs an appropriate eligibility
    policy and load/force feedback instead of trusting a class label.
    """
    if det.kind != "debris" or det.object_class not in LITTER_CLASSES:
        return "unrecognized litter class"
    if not np.isfinite(det.position_robot).all():
        return "invalid position"
    if det.size_m is None or det.estimated_mass_kg is None:
        return "dimensions or payload estimate unavailable"
    if len(det.size_m) != 3:
        return "invalid dimensions"
    length, width, height = det.size_m
    values = np.array([length, width, height, det.estimated_mass_kg])
    if not np.isfinite(values).all() or np.any(values <= 0):
        return "invalid dimensions or payload estimate"
    if det.orientation_rad is not None and not np.isfinite(det.orientation_rad):
        return "invalid orientation"
    if length > cfg.max_length_m or width > min(cfg.max_width_m, cfg.opening_width_m):
        return "item does not fit collector"
    if height > cfg.max_height_m:
        return "item is too tall"
    if det.estimated_mass_kg > cfg.max_payload_kg:
        return "payload exceeds limit"
    return None


def collector_position(pose_xytheta, cfg: CollectorConfig, carrying=False) -> np.ndarray:
    x, y, theta = pose_xytheta
    offset = cfg.carry_offset_m if carrying else cfg.pickup_offset_m
    return np.array([x + offset * np.cos(theta), y + offset * np.sin(theta)])


def collector_navigation_radius(cfg: CollectorConfig, chassis_radius: float,
                                extended: bool) -> float:
    """Conservative swept radius; does not change the mop cleaning radius."""
    if not extended:
        return max(chassis_radius, cfg.stowed_front_m)
    front = cfg.carry_offset_m + max(cfg.palm_depth_m, cfg.max_length_m) / 2
    return max(chassis_radius, float(np.hypot(front, cfg.opening_width_m / 2)))
