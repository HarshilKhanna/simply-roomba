"""
perception.py
--------------
RGB-D perception stack for the cleaning robot. Pure NumPy/SciPy core so it
is unit-testable without ROS, OpenCV, or Open3D running. The ROS node
(cleaning_robot_node.py) is a thin wrapper that feeds this module real
sensor data and republishes results.

Design summary
==============
1. GroundPlaneEstimator
   - Back-projects the depth image to a camera-frame point cloud.
   - Fits the floor plane with RANSAC on a subsample (fast, robust to the
     debris/stain outliers we're trying to detect), refined with an SVD
     least-squares pass on the inlier set.
   - Refit is amortized (every N frames) since the camera-to-floor
     geometry is rigid; per-frame work is just a dot product against the
     cached plane (a fast height-map, not a fresh RANSAC every frame).

2. DebrisDetector
   - Thresholds the per-pixel signed height above the cached plane
     (5mm < h < ~15cm) to get a binary mask.
   - Connected-component labeling (image-space, O(HW)) replaces a full 3D
     clustering step (e.g. DBSCAN) because the depth image is already an
     organized grid -- this is the "fast point cloud filtering" approach
     rather than per-point Euclidean clustering.

3. FloorColorModel / StainDetector
   - Maintains a running Gaussian color model (mean + covariance in CIE
     Lab) of *clean* floor, updated with an EMA and robust outlier
     rejection so stains don't bleed into the background model.
   - Per-pixel Mahalanobis distance against that model flags color
     anomalies; combined with the depth-plane "flatness" mask (anomaly is
     co-planar with the floor, not raised) to separate stains from debris
     and from shadows/edges that also perturb RGB.

4. CoordinateTransformer
   - Depth-based back-projection for debris (reliable return).
   - Ray/plane intersection for stains (liquids are frequently
     IR-transparent or specular, giving noisy/absent depth returns right
     where you need them -- so stains are localized by intersecting the
     pixel's viewing ray with the *known floor plane* instead of trusting
     per-pixel depth).
   - Static extrinsic transform from camera optical frame to robot
     base_link.

5. DetectionTracker
   - Nearest-neighbor association across frames + a "confirm after N
     consecutive detections" gate, so a single noisy frame can't send the
     robot chasing a shadow.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .config import CameraIntrinsics, CameraExtrinsics, PerceptionConfig


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def depth_image_to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
    """Normalize ROS depth encodings, not native SDK Z16 depth units."""
    if encoding == "16UC1":
        result = np.asarray(depth, dtype=np.float32) * 0.001
    elif encoding == "32FC1":
        result = np.asarray(depth, dtype=np.float32).copy()
    else:
        raise ValueError(f"unsupported ROS depth encoding: {encoding}")
    result[~np.isfinite(result) | (result <= 0)] = 0
    return result


def interpolate_planar_pose(samples, timestamp: float, max_extrapolation_s=0.1):
    """Interpolate timestamped (time, x/y/yaw) odometry across yaw wrapping."""
    if not samples:
        return None
    if timestamp <= samples[0][0]:
        return samples[0][1] if samples[0][0] - timestamp <= max_extrapolation_s else None
    if timestamp >= samples[-1][0]:
        return samples[-1][1] if timestamp - samples[-1][0] <= max_extrapolation_s else None
    for (before, pose_before), (after, pose_after) in zip(samples, samples[1:]):
        if before <= timestamp <= after:
            alpha = (timestamp - before) / max(after - before, 1e-12)
            xy = (1-alpha) * np.asarray(pose_before[:2]) + alpha * np.asarray(pose_after[:2])
            yaw_delta = (pose_after[2] - pose_before[2] + np.pi) % (2*np.pi) - np.pi
            yaw = pose_before[2] + alpha*yaw_delta
            return float(xy[0]), float(xy[1]), float((yaw + np.pi) % (2*np.pi) - np.pi)
    return None


def backproject_depth(depth_m: np.ndarray, intr: CameraIntrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized pinhole back-projection.

    Args:
        depth_m: (H, W) float32 depth image in meters, 0 = invalid.
        intr: camera intrinsics.

    Returns:
        points: (H, W, 3) camera-optical-frame XYZ (x-right, y-down, z-fwd).
        valid: (H, W) bool mask of pixels with usable depth.
    """
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_m
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    points = np.stack([x, y, z], axis=-1)
    valid = np.isfinite(z) & (z > 1e-4)
    return points, valid


def fit_plane_ransac(points: np.ndarray, iterations: int, inlier_thresh: float,
                      rng: Optional[np.random.Generator] = None
                      ) -> Optional[Tuple[np.ndarray, float, float]]:
    """RANSAC plane fit with an SVD refinement pass.

    Args:
        points: (N, 3) candidate points (already subsampled for speed).
        iterations: RANSAC trials.
        inlier_thresh: max point-to-plane distance (m) to count as inlier.

    Returns:
        (normal, d, inlier_ratio) such that normal . p + d == 0 on the
        plane, with `normal` oriented toward the camera origin (so that
        objects between the camera and the floor -- i.e. sticking up off
        the floor -- have POSITIVE signed height). None if fit fails.
    """
    n = points.shape[0]
    if n < 20:
        return None
    rng = rng or np.random.default_rng()

    best_count = -1
    best_inliers = None
    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -normal.dot(p1)
        dist = np.abs(points @ normal + d)
        inliers = dist < inlier_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 10:
        return None

    # Refine with SVD least-squares over all inliers (more accurate than
    # the 3-point sample alone).
    inlier_pts = points[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_pts - centroid)
    normal = vt[-1]
    d = -normal.dot(centroid)

    # Orient normal so the camera origin (0,0,0) is on the positive side.
    if normal.dot(np.zeros(3)) + d < 0:
        normal = -normal
        d = -d

    inlier_ratio = best_count / n
    return normal, float(d), inlier_ratio


def signed_height(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """Signed distance of each point from the plane; positive = above floor
    (toward the camera side), by the orientation convention in fit_plane_ransac."""
    return points @ normal + d


def intersect_ray_plane(u: np.ndarray, v: np.ndarray, intr: CameraIntrinsics,
                         normal: np.ndarray, d: float) -> np.ndarray:
    """Intersect the camera ray through pixel(s) (u, v) with the floor plane.

    Used for stains where the per-pixel depth return is unreliable (wet /
    specular / IR-transparent surfaces). u, v may be scalars or arrays.

    Returns camera-frame XYZ, shape (..., 3).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    # Ray direction in camera optical frame for a pixel at unit depth.
    dir_x = (u - intr.cx) / intr.fx
    dir_y = (v - intr.cy) / intr.fy
    dir_z = np.ones_like(u)
    ray = np.stack([dir_x, dir_y, dir_z], axis=-1)

    denom = ray @ normal
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    t = -d / denom
    t = np.clip(t, 0.0, None)  # discard intersections behind the camera
    return ray * t[..., None]


def transform_points(points_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an array of camera-frame points
    of shape (..., 3), returning robot-base-frame points of the same shape."""
    shape = points_cam.shape
    flat = points_cam.reshape(-1, 3)
    homog = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    out = (T_base_cam @ homog.T).T[:, :3]
    return out.reshape(shape)


# ----------------------------------------------------------------------------
# Ground plane estimator (stateful, amortized RANSAC)
# ----------------------------------------------------------------------------

class GroundPlaneEstimator:
    def __init__(self, cfg: PerceptionConfig, intr: CameraIntrinsics,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.intr = intr
        self.rng = rng or np.random.default_rng()
        self.normal: Optional[np.ndarray] = None
        self.d: Optional[float] = None
        self._frame_count = 0

    @property
    def is_valid(self) -> bool:
        return self.normal is not None

    def update(self, points: np.ndarray, valid_mask: np.ndarray) -> bool:
        """Call once per frame. Internally amortizes the RANSAC refit.
        Returns True if the plane estimate is (now) valid."""
        need_refit = (
            self.normal is None
            or self._frame_count % self.cfg.plane_refit_every_n_frames == 0
        )
        self._frame_count += 1

        if need_refit:
            valid_pts = points[valid_mask]
            if valid_pts.shape[0] < 50:
                return self.is_valid
            n_sample = min(self.cfg.plane_sample_points, valid_pts.shape[0])
            sample_idx = self.rng.choice(valid_pts.shape[0], size=n_sample, replace=False)
            sample = valid_pts[sample_idx]
            result = fit_plane_ransac(
                sample, self.cfg.plane_ransac_iterations,
                self.cfg.plane_ransac_inlier_thresh_m, self.rng)
            if result is not None:
                normal, d, inlier_ratio = result
                # Guard against a bad fit clobbering a good previous one
                # (e.g. a frame full of an open doorway with no floor).
                if inlier_ratio > 0.4:
                    self.normal, self.d = normal, d
        return self.is_valid

    def height_map(self, points: np.ndarray) -> np.ndarray:
        assert self.is_valid, "call update() with a valid frame first"
        return signed_height(points, self.normal, self.d)


# ----------------------------------------------------------------------------
# Detections
# ----------------------------------------------------------------------------

@dataclass(eq=False)
class Detection:
    kind: str                      # "debris" | "stain"
    position_robot: np.ndarray     # (x, y, z) meters, robot base_link frame
    radius_m: float                # approx footprint radius, for planning
    pixel_centroid: Tuple[int, int]
    frame_id: int = 0
    track_id: Optional[int] = None
    confidence: int = 1
    # position_robot is retained for API compatibility. Its actual frame is
    # explicit here; the mission planner consumes "odom"/"planning" positions.
    coordinate_frame: str = "base_link"
    object_class: Optional[str] = None
    size_m: Optional[Tuple[float, float, float]] = None  # length, width, height
    orientation_rad: Optional[float] = None
    estimated_mass_kg: Optional[float] = None
    observed_at_s: Optional[float] = None


def detection_to_planning_frame(det: Detection, pose_xytheta,
                                frame: str = "odom") -> Detection:
    """Copy a base-relative observation into a fixed planning frame.

    Supply the robot pose at the observation timestamp, not a later control
    pose. Already-fixed observations are copied, never transformed twice.
    """
    if det.coordinate_frame in (frame, "planning"):
        return replace(det, position_robot=np.array(det.position_robot, copy=True))
    if det.coordinate_frame != "base_link":
        raise ValueError(f"cannot plan a detection in {det.coordinate_frame!r}")
    x, y, theta = pose_xytheta
    c, s = np.cos(theta), np.sin(theta)
    position = np.array(det.position_robot, dtype=float, copy=True)
    position[:2] = np.array([[c, -s], [s, c]]) @ position[:2] + [x, y]
    orientation = None if det.orientation_rad is None else det.orientation_rad + theta
    return replace(det, position_robot=position, coordinate_frame=frame,
                   orientation_rad=orientation)


class DebrisDetector:
    """Geometric proposals for pixels raised above the floor plane.

    These unclassified detections remain obstacles. A semantic/eligibility
    adapter must supply litter class and dimensions before the collector
    mission can select them; height alone does not identify disposable litter.
    """

    def __init__(self, cfg: PerceptionConfig, intr: CameraIntrinsics):
        self.cfg = cfg
        self.intr = intr

    def detect(self, points_cam: np.ndarray, valid_mask: np.ndarray,
               height_map: np.ndarray) -> List[Detection]:
        c = self.cfg
        range_m = points_cam[..., 2]
        mask = (
            valid_mask
            & (height_map > c.debris_min_height_m)
            & (height_map < c.debris_max_height_m)
            & (range_m > c.debris_min_range_m)
            & (range_m < c.debris_max_range_m)
        )
        if not mask.any():
            return []

        labeled, n_labels = ndimage.label(mask)
        detections: List[Detection] = []
        for label_id in range(1, n_labels + 1):
            comp_mask = labeled == label_id
            area = int(comp_mask.sum())
            if area < c.debris_min_pixel_area:
                continue
            pts = points_cam[comp_mask]
            centroid_cam = pts.mean(axis=0)
            # Rough planar extent of the cluster, for obstacle-radius sizing.
            xy = pts[:, :2] - centroid_cam[:2]
            extent = float(np.sqrt((xy ** 2).sum(axis=1)).max()) if len(xy) else 0.02
            vs, us = np.nonzero(comp_mask)
            pixel_centroid = (int(us.mean()), int(vs.mean()))
            detections.append(Detection(
                kind="debris",
                position_robot=centroid_cam,  # transformed to robot frame by caller
                radius_m=extent,
                pixel_centroid=pixel_centroid,
                coordinate_frame="camera_optical",
            ))
        return detections


class FloorColorModel:
    """Running Gaussian (mean + covariance) of clean-floor color in Lab
    space, robustly updated so stains don't get absorbed into 'normal'."""

    def __init__(self, cfg: PerceptionConfig):
        self.cfg = cfg
        self.mean: Optional[np.ndarray] = None
        self.cov: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None

    @property
    def is_initialized(self) -> bool:
        return self.mean is not None

    def initialize(self, lab_pixels: np.ndarray) -> None:
        """lab_pixels: (N, 3) sample of *known-clean* floor, e.g. gathered
        during a startup calibration sweep."""
        self.mean = lab_pixels.mean(axis=0)
        self.cov = np.cov(lab_pixels, rowvar=False) + np.eye(3) * 1e-3
        self._refresh_inverse()

    def update(self, lab_pixels: np.ndarray) -> None:
        """EMA update using only the pixels that currently look 'floor-like'
        (caller passes floor-plane, non-flagged pixels); additionally drops
        the most extreme percentile so a not-yet-detected stain can't drag
        the model toward itself."""
        if lab_pixels.shape[0] < 50:
            return
        if not self.is_initialized:
            self.initialize(lab_pixels)
            return

        d = lab_pixels - self.mean
        mahal = np.einsum('ij,jk,ik->i', d, self.inv_cov, d)
        keep_thresh = np.percentile(mahal, self.cfg.color_model_outlier_percentile)
        robust_pixels = lab_pixels[mahal <= keep_thresh]
        if robust_pixels.shape[0] < 20:
            return

        alpha = self.cfg.color_model_ema_alpha
        batch_mean = robust_pixels.mean(axis=0)
        batch_cov = np.cov(robust_pixels, rowvar=False) + np.eye(3) * 1e-3
        self.mean = (1 - alpha) * self.mean + alpha * batch_mean
        self.cov = (1 - alpha) * self.cov + alpha * batch_cov
        self._refresh_inverse()

    def mahalanobis_map(self, lab_image: np.ndarray) -> np.ndarray:
        """lab_image: (H, W, 3) -> (H, W) squared Mahalanobis distance."""
        h, w, _ = lab_image.shape
        flat = lab_image.reshape(-1, 3)
        d = flat - self.mean
        mahal_sq = np.einsum('ij,jk,ik->i', d, self.inv_cov, d)
        return mahal_sq.reshape(h, w)

    def _refresh_inverse(self) -> None:
        self.inv_cov = np.linalg.inv(self.cov)


class StainDetector:
    """Flat 2D color-anomaly detection (stains, spills) via Mahalanobis
    distance against a learned floor color model, gated to pixels that are
    co-planar with the floor (so raised debris and its cast shadow don't
    get double-counted as a stain)."""

    def __init__(self, cfg: PerceptionConfig, color_model: FloorColorModel):
        self.cfg = cfg
        self.color_model = color_model

    def floor_plane_mask(self, height_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        c = self.cfg
        return valid_mask & (np.abs(height_map) < c.stain_flatness_band_m)

    def detect(self, lab_image: np.ndarray, floor_mask: np.ndarray) -> List[Detection]:
        if not self.color_model.is_initialized:
            return []
        c = self.cfg
        mahal_sq = self.color_model.mahalanobis_map(lab_image)
        candidate = (mahal_sq > c.mahalanobis_threshold ** 2) & floor_mask

        if not candidate.any():
            return []

        # Clean up speckle noise / thin edges before connected-components.
        k = c.stain_morph_kernel
        struct = np.ones((k, k), dtype=bool)
        candidate = ndimage.binary_opening(candidate, structure=struct)
        candidate = ndimage.binary_closing(candidate, structure=struct)

        labeled, n_labels = ndimage.label(candidate)
        detections: List[Detection] = []
        for label_id in range(1, n_labels + 1):
            comp_mask = labeled == label_id
            area = int(comp_mask.sum())
            if area < c.stain_min_pixel_area:
                continue
            vs, us = np.nonzero(comp_mask)
            pixel_centroid = (int(us.mean()), int(vs.mean()))
            # Pixel-space extent -> used later once we know metric scale
            # per-pixel from the plane intersection (done by the caller /
            # CoordinateTransformer, since it needs the plane + intrinsics).
            pixel_extent = float(np.sqrt(
                (us - us.mean()) ** 2 + (vs - vs.mean()) ** 2
            ).max())
            detections.append(Detection(
                kind="stain",
                position_robot=np.array([np.nan, np.nan, np.nan]),  # filled in by caller
                radius_m=pixel_extent,  # placeholder in PIXELS, replaced by caller
                pixel_centroid=pixel_centroid,
            ))
        return detections

    def update_background_model(self, lab_image: np.ndarray, floor_mask: np.ndarray,
                                 stain_mask: Optional[np.ndarray] = None) -> None:
        mask = floor_mask if stain_mask is None else (floor_mask & ~stain_mask)
        pixels = lab_image[mask]
        self.color_model.update(pixels)


# ----------------------------------------------------------------------------
# Coordinate transformer
# ----------------------------------------------------------------------------

class CoordinateTransformer:
    def __init__(self, intr: CameraIntrinsics, extr: CameraExtrinsics):
        self.intr = intr
        self.T_base_cam = extr.transform_matrix()

    def debris_to_robot_frame(self, detections: List[Detection],
                               points_cam: np.ndarray) -> List[Detection]:
        """Debris detections already carry a reliable depth-based centroid
        in `position_robot` (still camera frame at this point) -- transform it."""
        out = []
        for det in detections:
            p_robot = transform_points(det.position_robot.reshape(1, 3), self.T_base_cam)[0]
            det.position_robot = p_robot
            det.coordinate_frame = "base_link"
            out.append(det)
        return out

    def stain_to_robot_frame(self, detections: List[Detection],
                              plane_normal: np.ndarray, plane_d: float) -> List[Detection]:
        """Stains use ray/plane intersection at the pixel centroid (robust
        to bad depth on liquid), then convert the pixel-space radius to a
        metric radius using the local plane distance."""
        out = []
        for det in detections:
            u, v = det.pixel_centroid
            p_cam = intersect_ray_plane(u, v, self.intr, plane_normal, plane_d)
            p_robot = transform_points(p_cam.reshape(1, 3), self.T_base_cam)[0]

            # Convert the pixel-space extent (radius_m field currently holds
            # PIXELS, set by StainDetector.detect) to meters using the local
            # ground-sample-distance implied by the ray/plane range.
            range_m = float(np.linalg.norm(p_cam))
            pixel_to_meter = range_m / self.intr.fx  # small-angle approx, per-pixel
            det.radius_m = max(det.radius_m * pixel_to_meter, 0.02)
            det.position_robot = p_robot
            det.coordinate_frame = "base_link"
            out.append(det)
        return out


# ----------------------------------------------------------------------------
# Temporal detection tracking / confirmation
# ----------------------------------------------------------------------------

class DetectionTracker:
    """Nearest-neighbor tracker with a 'confirm after N consecutive frames'
    gate, so a single-frame false positive can't trigger a divert."""

    def __init__(self, cfg: PerceptionConfig, max_missed_frames: Optional[int] = 5):
        self.cfg = cfg
        self.max_missed_frames = max_missed_frames
        self._tracks: List[Detection] = []
        self._next_id = 0
        self._misses = {}  # track_id -> consecutive miss count

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update ONCE per sensor frame with all kinds, in one fixed frame.

        Each track can receive at most one hit per frame. Missing a frame
        resets its consecutive-hit count; retained tracks still keep their
        identity for up to five missed frames by default. max_missed_frames=None
        retains fixed-world track identities until explicitly removed.
        """
        gate = self.cfg.detection_match_radius_m
        matched_track_ids = set()
        previous_tracks = list(self._tracks)

        for det in detections:
            best_track = None
            best_dist = gate
            for track in previous_tracks:
                if id(track) in matched_track_ids or track.kind != det.kind:
                    continue
                if track.coordinate_frame != det.coordinate_frame:
                    continue
                if (track.object_class is not None and det.object_class is not None
                        and track.object_class != det.object_class):
                    continue
                dist = float(np.linalg.norm(
                    track.position_robot[:2] - det.position_robot[:2]))
                if dist < best_dist:
                    best_dist = dist
                    best_track = track

            if best_track is not None:
                # Exponential smoothing on position, increment confidence.
                best_track.position_robot = 0.7 * best_track.position_robot + 0.3 * det.position_robot
                best_track.radius_m = max(best_track.radius_m, det.radius_m)
                best_track.confidence += 1
                best_track.pixel_centroid = det.pixel_centroid
                best_track.object_class = det.object_class
                best_track.size_m = det.size_m
                best_track.orientation_rad = det.orientation_rad
                best_track.estimated_mass_kg = det.estimated_mass_kg
                best_track.observed_at_s = det.observed_at_s
                matched_track_ids.add(id(best_track))
                self._misses[id(best_track)] = 0
            else:
                det.track_id = self._next_id
                self._next_id += 1
                self._tracks.append(det)
                matched_track_ids.add(id(det))
                self._misses[id(det)] = 0

        # Decay/prune tracks that weren't matched this frame.
        surviving = []
        for track in self._tracks:
            if id(track) not in matched_track_ids:
                self._misses[id(track)] = self._misses.get(id(track), 0) + 1
                track.confidence = 0
                if (self.max_missed_frames is not None
                        and self._misses[id(track)] > self.max_missed_frames):
                    self._misses.pop(id(track), None)
                    continue  # drop stale track
            surviving.append(track)
        self._tracks = surviving

        return [t for t in self._tracks
                if t.confidence >= self.cfg.min_consecutive_detections]

    def remove(self, track: Detection) -> None:
        """Call after a stain has been scrubbed / debris avoided, so it's
        not immediately re-triggered."""
        self._tracks = [t for t in self._tracks if t is not track]
        self._misses.pop(id(track), None)
