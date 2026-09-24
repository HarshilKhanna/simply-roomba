"""
Sanity tests against SYNTHETIC data (no ROS / camera / robot required).
Run: python3 test_pipeline.py
"""
import numpy as np
from cleaning_robot.config import RobotConfig
from cleaning_robot.perception import (
    backproject_depth, fit_plane_ransac, signed_height, intersect_ray_plane,
    GroundPlaneEstimator, DebrisDetector, FloorColorModel, StainDetector,
    CoordinateTransformer, DetectionTracker, Detection, transform_points,
)
from cleaning_robot.motion_planning import (
    generate_spot_scrub_path, generate_boustrophedon, PurePursuitController,
    MissionPlanner, MissionState,
)

cfg = RobotConfig()
rng = np.random.default_rng(0)


def make_synthetic_frame(debris_boxes=(), stain_boxes=(), noise=0.0005):
    """Build a synthetic depth+rgb frame via dense analytic ray/plane
    intersection (every pixel gets a value -- no sampling holes), then
    locally raises the plane inside debris boxes. This mirrors what a real
    depth camera returns (dense) far better than forward-projecting a
    sparse world-space grid would.
    """
    h, w = cfg.intrinsics.height, cfg.intrinsics.width
    intr = cfg.intrinsics
    extr = cfg.extrinsics
    T_base_cam = extr.transform_matrix()
    R_base_cam = T_base_cam[:3, :3]
    t_base_cam = T_base_cam[:3, 3]

    normal_base = np.array([0.0, 0.0, 1.0])
    normal_cam = R_base_cam.T @ normal_base
    d_cam_floor = normal_base.dot(t_base_cam)  # d_cam for bump=0

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    def intersect(d_cam):
        return intersect_ray_plane(u, v, intr, normal_cam, d_cam)

    p_cam0 = intersect(d_cam_floor)  # (H, W, 3), bump=0 everywhere
    p_robot0 = transform_points(p_cam0, T_base_cam)  # (H, W, 3), z~0

    rgb_lab = np.zeros((h, w, 3), dtype=np.float32)
    rgb_lab[..., 0] = 70.0
    rgb_lab[..., 1] = 2.0
    rgb_lab[..., 2] = 4.0

    bump_m = np.zeros((h, w), dtype=np.float64)
    for (bx0, bx1, by0, by1, h_mm) in debris_boxes:
        in_box = ((p_robot0[..., 0] >= bx0) & (p_robot0[..., 0] <= bx1) &
                  (p_robot0[..., 1] >= by0) & (p_robot0[..., 1] <= by1))
        bump_m[in_box] = h_mm / 1000.0

    for (sx0, sx1, sy0, sy1) in stain_boxes:
        in_box = ((p_robot0[..., 0] >= sx0) & (p_robot0[..., 0] <= sx1) &
                  (p_robot0[..., 1] >= sy0) & (p_robot0[..., 1] <= sy1))
        rgb_lab[in_box] = [45.0, 18.0, -22.0]  # coffee-ish stain in Lab

    d_cam_pixel = d_cam_floor - bump_m  # per-pixel plane offset for bumps
    ray = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u, dtype=float)], axis=-1)
    denom = ray @ normal_cam
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    t = -d_cam_pixel / denom
    t = np.clip(t, 0.0, None)
    p_cam = ray * t[..., None]

    depth = p_cam[..., 2].astype(np.float32)
    depth += rng.normal(0, noise, size=depth.shape).astype(np.float32)
    depth[depth <= 0] = 0.0
    return depth, rgb_lab


def test_plane_fit_and_debris():
    print("== Plane fit + debris detection ==")
    depth, lab = make_synthetic_frame(debris_boxes=[(0.6, 0.65, -0.05, 0.0, 15.0)])
    points, valid = backproject_depth(depth, cfg.intrinsics)

    plane_est = GroundPlaneEstimator(cfg.perception, cfg.intrinsics, rng)
    ok = plane_est.update(points, valid)
    assert ok, "plane fit failed"

    # Compare fitted plane normal to ground truth (floor plane in camera
    # frame derived from the extrinsics).
    T_base_cam = cfg.extrinsics.transform_matrix()
    R_base_cam = T_base_cam[:3, :3]
    true_normal_base = np.array([0, 0, 1.0])
    true_normal_cam = R_base_cam.T @ true_normal_base
    true_normal_cam /= np.linalg.norm(true_normal_cam)
    cos_angle = abs(np.dot(plane_est.normal, true_normal_cam))
    print(f"  plane normal alignment cos(angle) = {cos_angle:.4f} (want ~1.0)")
    assert cos_angle > 0.98

    hmap = plane_est.height_map(points)
    debris_detector = DebrisDetector(cfg.perception, cfg.intrinsics)
    dets = debris_detector.detect(points, valid, hmap)
    print(f"  debris clusters found: {len(dets)}")
    assert len(dets) >= 1, "expected to detect the synthetic 15mm debris patch"

    transformer = CoordinateTransformer(cfg.intrinsics, cfg.extrinsics)
    dets = transformer.debris_to_robot_frame(dets, points)
    for d in dets:
        print(f"  debris @ robot frame = {np.round(d.position_robot, 3)} (truth ~x=0.625,y=-0.025,z=0.015)")
    print("  OK\n")


def test_stain_detection_and_localization():
    print("== Stain color detection + ray/plane localization ==")
    depth, lab = make_synthetic_frame(stain_boxes=[(1.0, 1.2, -0.1, 0.1)])
    points, valid = backproject_depth(depth, cfg.intrinsics)

    plane_est = GroundPlaneEstimator(cfg.perception, cfg.intrinsics, rng)
    plane_est.update(points, valid)
    hmap = plane_est.height_map(points)

    color_model = FloorColorModel(cfg.perception)
    # Calibrate on a patch of known-clean floor pixels.
    clean_sample = lab[valid & (np.abs(hmap) < 0.005)]
    clean_sample = clean_sample[np.random.default_rng(1).choice(len(clean_sample), 2000, replace=False)]
    color_model.initialize(clean_sample)

    stain_detector = StainDetector(cfg.perception, color_model)
    floor_mask = stain_detector.floor_plane_mask(hmap, valid)
    dets = stain_detector.detect(lab, floor_mask)
    print(f"  stain blobs found: {len(dets)}")
    assert len(dets) >= 1, "expected to detect the synthetic stain patch"

    transformer = CoordinateTransformer(cfg.intrinsics, cfg.extrinsics)
    dets = transformer.stain_to_robot_frame(dets, plane_est.normal, plane_est.d)
    for d in dets:
        print(f"  stain @ robot frame = {np.round(d.position_robot, 3)}, "
              f"radius={d.radius_m:.3f} m (truth center ~x=1.1,y=0.0, half-width~0.1m)")
    print("  OK\n")


def test_spiral_coverage_guarantee():
    print("== Spiral scrub path coverage guarantee ==")
    R = cfg.geometry.radius_m
    overlap = cfg.planning.spiral_overlap
    stain_r = 0.35
    margin = cfg.planning.spiral_margin_m
    target_radius = stain_r + margin  # the area we actually promised to cover
    path = generate_spot_scrub_path(R, stain_r, overlap, margin)
    radii = np.linalg.norm(path, axis=1)
    print(f"  path points: {len(path)}, spiral tip radius: {radii.max():.3f}, "
          f"target radius: {target_radius:.3f}")
    assert radii.max() >= target_radius

    # Monte Carlo coverage check over the TARGET disk (the spiral's tip may
    # slightly overshoot this due to full-turn rounding -- that overshoot
    # is a harmless bonus, not something we need to re-cover).
    rng2 = np.random.default_rng(2)
    test_pts = rng2.uniform(-target_radius, target_radius, size=(20000, 2))
    test_pts = test_pts[np.linalg.norm(test_pts, axis=1) <= target_radius]
    from scipy.spatial import cKDTree
    tree = cKDTree(path)
    dists, _ = tree.query(test_pts)
    covered = (dists <= R).mean()
    print(f"  fraction of target disk covered by mop footprint union: {covered*100:.3f}%")
    assert covered > 0.995, "spiral pitch does not guarantee full coverage"
    print("  OK\n")


def test_boustrophedon_and_pure_pursuit():
    print("== Boustrophedon + pure pursuit following ==")
    path = generate_boustrophedon(0, 2.0, 0, 1.0, cfg.geometry.radius_m, cfg.planning.coverage_lane_overlap)
    print(f"  lawnmower waypoints: {len(path)}")
    assert len(path) > 2

    controller = PurePursuitController(cfg.planning, cfg.geometry)
    pose = (0.0, 0.0, 0.0)
    idx = 0
    steps = 0
    finished = False
    while not finished and steps < 20000:
        v, w, idx, finished = controller.step(pose, path, idx, cfg.geometry.max_linear_vel)
        dt = 0.05
        x, y, theta = pose
        theta += w * dt
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        pose = (x, y, theta)
        steps += 1
    print(f"  reached end of coverage path in {steps} steps ({steps*0.05:.1f}s sim time), finished={finished}")
    assert finished
    print("  OK\n")


def test_mission_state_machine():
    print("== Full mission state machine: coverage -> divert -> scrub -> resume ==")
    path = generate_boustrophedon(0, 1.5, 0, 0.8, cfg.geometry.radius_m, cfg.planning.coverage_lane_overlap)
    planner = MissionPlanner(cfg.planning, cfg.geometry, path)

    stain = Detection(kind="stain", position_robot=np.array([0.5, 0.3, 0.0]),
                       radius_m=0.06, pixel_centroid=(0, 0), track_id=1, confidence=5)

    pose = (0.0, 0.0, 0.0)
    saw_diverting = saw_scrubbing = resumed_coverage = False
    dt = 0.05
    stains_this_frame = [stain]
    for step in range(20000):
        v, w = planner.update(pose, stains_this_frame, [])
        stains_this_frame = []  # only announce once; tracker persistence is external to this test
        x, y, theta = pose
        theta += w * dt
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        pose = (x, y, theta)

        if planner.state == MissionState.DIVERTING:
            saw_diverting = True
        if planner.state == MissionState.SCRUBBING:
            saw_scrubbing = True
        if saw_scrubbing and planner.state == MissionState.COVERAGE:
            resumed_coverage = True
            break
        if planner.state == MissionState.DONE:
            break

    print(f"  visited DIVERTING={saw_diverting}, SCRUBBING={saw_scrubbing}, resumed COVERAGE={resumed_coverage}")
    assert saw_diverting and saw_scrubbing and resumed_coverage
    assert 1 in planner.handled_track_ids
    print("  OK\n")


if __name__ == "__main__":
    test_plane_fit_and_debris()
    test_stain_detection_and_localization()
    test_spiral_coverage_guarantee()
    test_boustrophedon_and_pure_pursuit()
    test_mission_state_machine()
    print("ALL TESTS PASSED")
