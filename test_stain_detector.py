"""
test_stain_detector.py
=======================
Interactive visual test tool for Step 1:
Validates the Mahalanobis Color-Anomaly Stain Detector on real images or live camera.

Usage:
  python test_stain_detector.py                     # Try webcam, fallback to synthetic
  python test_stain_detector.py --camera 0          # Use webcam 0
  python test_stain_detector.py --image floor.jpg   # Use static image

Controls:
  Drag Mouse  : Select a clean floor rectangle to calibrate (or press [C])
  [C]         : Calibrate background model using current ROI or center patch
  [U]         : Toggle online background updating (EMA)
  [R]         : Reset calibration
  [S]         : Save visual screenshot (input, heatmap, mask, detections)
  [SPACE]     : Pause / Resume video
  [Q] / [ESC] : Quit
"""

import sys
import os
import argparse
import time
import numpy as np
import cv2

# Add cleaning_robot to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cleaning_robot.config import PerceptionConfig
from cleaning_robot.perception import FloorColorModel, StainDetector, Detection


def rgb_to_cielab(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to standard CIE Lab (L: 0..100, a: -128..127, b: -128..127)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= (100.0 / 255.0)
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


def create_synthetic_floor(w=640, h=480) -> np.ndarray:
    """Creates a realistic synthetic tiled floor with lighting gradient,
    grout lines, shadows, and spilled coffee/stain anomalies."""
    rng = np.random.default_rng(42)
    # Clean floor base color: warm beige tile
    base_bgr = np.array([170, 195, 210], dtype=np.float32)
    img = np.ones((h, w, 3), dtype=np.float32) * base_bgr

    # Texture noise
    img += rng.normal(0, 3.5, (h, w, 3))

    # Vignetting / lighting gradient across room
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    light_factor = 1.0 - 0.25 * (xx**2 + yy**2) + 0.1 * xx
    img *= light_factor[..., None]

    # Grout grid lines (subtle lines)
    grid_size = 120
    img[::grid_size, :] *= 0.82
    img[:, ::grid_size] *= 0.82

    # Cast shadow across diagonal (testing shadow invariance in Lab space)
    shadow_mask = (yy - 0.5 * xx > 0.1) & (yy - 0.5 * xx < 0.6)
    img[shadow_mask] *= 0.70  # Drops luminance without shifting a/b significantly

    # Stain 1: Coffee spill (dark brown, high chroma shift)
    c1_x, c1_y, r1 = 200, 260, 32
    dist1 = np.sqrt((np.arange(w)[None, :] - c1_x)**2 + (np.arange(h)[:, None] - c1_y)**2)
    stain1_mask = dist1 <= r1
    img[stain1_mask] = np.array([40, 65, 110], dtype=np.float32) + rng.normal(0, 4, (int(stain1_mask.sum()), 3))

    # Stain 2: Faint colored juice / chemical drip (subtle)
    c2_x, c2_y, r2 = 450, 180, 22
    dist2 = np.sqrt((np.arange(w)[None, :] - c2_x)**2 + (np.arange(h)[:, None] - c2_y)**2)
    stain2_mask = dist2 <= r2
    img[stain2_mask] = np.array([130, 130, 215], dtype=np.float32)

    # Debris: small high-contrast crumb
    cv2.circle(img, (320, 350), 7, (20, 20, 20), -1)

    return np.clip(img, 0, 255).astype(np.uint8)


class InteractiveTester:
    def __init__(self, source_type="auto", source_arg=0):
        self.cfg = PerceptionConfig()
        self.color_model = FloorColorModel(self.cfg)
        self.stain_detector = StainDetector(self.cfg, self.color_model)

        self.source_type = source_type
        self.source_arg = source_arg
        self.cap = None
        self.static_frame = None

        # ROI selection state
        self.roi_pts = []
        self.selecting_roi = False
        self.roi_rect = None
        self.enable_ema = False
        self.paused = False

        self._init_source()

    def _init_source(self):
        if self.source_type == "image":
            if os.path.exists(str(self.source_arg)):
                self.static_frame = cv2.imread(str(self.source_arg))
                print(f"[+] Loaded image from {self.source_arg}")
            else:
                print(f"[-] Image not found at {self.source_arg}, falling back to synthetic floor.")
                self.static_frame = create_synthetic_floor()
        elif self.source_type == "camera":
            self.cap = cv2.VideoCapture(int(self.source_arg))
            if not self.cap.isOpened():
                print(f"[-] Camera {self.source_arg} not available, falling back to synthetic floor.")
                self.cap = None
                self.static_frame = create_synthetic_floor()
        else: # auto
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print("[+] Live webcam opened successfully.")
                else:
                    self.cap.release()
                    self.cap = None
                    self.static_frame = create_synthetic_floor()
            else:
                self.cap = None
                print("[*] No webcam detected. Using synthetic test environment.")
                self.static_frame = create_synthetic_floor()

    def on_mouse(self, event, x, y, flags, param):
        h, w = self.current_h, self.current_w
        if x > w or y > h:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting_roi = True
            self.roi_pts = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting_roi:
            if len(self.roi_pts) == 1:
                self.roi_pts = [self.roi_pts[0], (x, y)]
            elif len(self.roi_pts) == 2:
                self.roi_pts[1] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.selecting_roi:
            self.selecting_roi = False
            if len(self.roi_pts) == 2:
                p1, p2 = self.roi_pts
                x0, x1 = sorted([p1[0], p2[0]])
                y0, y1 = sorted([p1[1], p2[1]])
                if (x1 - x0) > 10 and (y1 - y0) > 10:
                    self.roi_rect = (x0, y0, x1, y1)
                    self.calibrate_from_roi()

    def calibrate_from_roi(self, rect=None):
        if rect is None:
            rect = self.roi_rect
        if rect is None:
            h, w = self.current_h, self.current_w
            rect = (w // 3, h // 3, 2 * w // 3, 2 * h // 3)

        x0, y0, x1, y1 = rect
        patch = self.current_lab[y0:y1, x0:x1]
        flat_patch = patch.reshape(-1, 3)

        self.color_model.initialize(flat_patch)
        mean_l, mean_a, mean_b = self.color_model.mean
        std_l = np.sqrt(self.color_model.cov[0, 0])
        std_a = np.sqrt(self.color_model.cov[1, 1])
        std_b = np.sqrt(self.color_model.cov[2, 2])
        print(f"[+] Calibrated clean floor model on {flat_patch.shape[0]} pixels:")
        print(f"    Mean Lab = [{mean_l:.1f}, {mean_a:.1f}, {mean_b:.1f}]")
        print(f"    Std  Lab = [{std_l:.2f}, {std_a:.2f}, {std_b:.2f}]")

    def run(self):
        window_name = "Step 1: Mahalanobis Stain Detector Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 800)
        cv2.setMouseCallback(window_name, self.on_mouse)

        # Trackbars for real-time parameter tuning
        cv2.createTrackbar("Mahal Thresh x10", window_name, int(self.cfg.mahalanobis_threshold * 10), 150, lambda v: None)
        cv2.createTrackbar("Min Area (px)", window_name, self.cfg.stain_min_pixel_area, 1000, lambda v: None)
        cv2.createTrackbar("Morph Kernel", window_name, self.cfg.stain_morph_kernel, 15, lambda v: None)

        print("\n" + "=" * 65)
        print("  STEP 1: MAHALANOBIS COLOR-ANOMALY STAIN DETECTOR")
        print("=" * 65)
        print("  Controls:")
        print("    [Click & Drag] : Select clean floor patch to calibrate")
        print("    [C]            : Auto-calibrate using center box")
        print("    [U]            : Toggle online EMA background updates")
        print("    [S]            : Save screenshot")
        print("    [SPACE]        : Pause / Resume")
        print("    [Q] / [ESC]    : Quit")
        print("=" * 65 + "\n")

        last_frame = None
        while True:
            if not self.paused or last_frame is None:
                if self.cap is not None:
                    ret, bgr = self.cap.read()
                    if not ret or bgr is None:
                        time.sleep(0.01)
                        continue
                else:
                    bgr = self.static_frame.copy()
                last_frame = bgr.copy()
            else:
                bgr = last_frame.copy()

            h, w = bgr.shape[:2]
            self.current_h, self.current_w = h, w

            # Read trackbar values
            thresh_val = max(cv2.getTrackbarPos("Mahal Thresh x10", window_name) / 10.0, 1.0)
            min_area = max(cv2.getTrackbarPos("Min Area (px)", window_name), 20)
            morph_k = cv2.getTrackbarPos("Morph Kernel", window_name)
            if morph_k % 2 == 0:
                morph_k += 1
            morph_k = max(morph_k, 1)

            self.cfg.mahalanobis_threshold = thresh_val
            self.cfg.stain_min_pixel_area = min_area
            self.cfg.stain_morph_kernel = morph_k

            # Convert to CIE Lab
            lab = rgb_to_cielab(bgr)
            self.current_lab = lab

            # Initial calibration if not done
            if not self.color_model.is_initialized:
                self.calibrate_from_roi()

            # 1. Compute Mahalanobis distance map
            mahal_sq = self.color_model.mahalanobis_map(lab)
            mahal_dist = np.sqrt(np.maximum(mahal_sq, 0.0))

            # 2. Stain detection (flatness assumed true for 2D test)
            floor_mask = np.ones((h, w), dtype=bool)
            dets = self.stain_detector.detect(lab, floor_mask)

            # Online EMA update if enabled
            if self.enable_ema and not self.paused:
                non_stain_mask = np.ones((h, w), dtype=bool)
                for d in dets:
                    u, v = d.pixel_centroid
                    r = int(d.radius_m) + 5
                    cv2.circle(non_stain_mask, (u, v), r, False, -1)
                update_sample = lab[non_stain_mask]
                if update_sample.shape[0] > 100:
                    self.color_model.update(update_sample)

            # --- Visualizations ---
            # Panel 1: Original image with overlays
            disp_overlay = bgr.copy()

            # Draw current ROI or calibration box
            if self.roi_rect:
                x0, y0, x1, y1 = self.roi_rect
                cv2.rectangle(disp_overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.putText(disp_overlay, "Calibration Box", (x0, max(y0 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Draw active mouse drag
            if self.selecting_roi and len(self.roi_pts) == 2:
                p1, p2 = self.roi_pts
                cv2.rectangle(disp_overlay, p1, p2, (0, 255, 0), 1)

            # Draw detected stains
            for i, d in enumerate(dets):
                u, v = d.pixel_centroid
                r = max(int(d.radius_m), 6)
                cv2.circle(disp_overlay, (u, v), r, (0, 0, 255), 2)
                cv2.drawMarker(disp_overlay, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
                label = f"Stain #{i+1} [r={r}px]"
                cv2.putText(disp_overlay, label, (u + 8, v - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Panel 2: Mahalanobis Distance Heatmap
            norm_dist = np.clip(mahal_dist / (thresh_val * 2.2), 0.0, 1.0)
            heatmap_gray = (norm_dist * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_TURBO)

            # Panel 3: Binary Candidate Mask (Thresholded)
            raw_mask = (mahal_dist > thresh_val).astype(np.uint8) * 255
            raw_mask_bgr = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)

            # Panel 4: Morphologically Cleaned Mask with Centroids
            clean_mask = np.zeros((h, w), dtype=np.uint8)
            for d in dets:
                u, v = d.pixel_centroid
                r = max(int(d.radius_m), 6)
                cv2.circle(clean_mask, (u, v), r, 255, -1)
            clean_mask_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)

            def add_title(img, text, color=(255, 255, 255)):
                cv2.rectangle(img, (0, 0), (w, 32), (30, 30, 30), -1)
                cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            add_title(disp_overlay, f"1. Detections ({len(dets)} stains found)", (0, 255, 255))
            add_title(heatmap_color, f"2. Mahalanobis Heatmap (Cutoff = {thresh_val:.1f} sigma)", (0, 200, 255))
            add_title(raw_mask_bgr, f"3. Raw Threshold Mask (D_M > {thresh_val:.1f})", (200, 200, 200))
            add_title(clean_mask_bgr, f"4. Morph Filter + MinArea ({min_area}px)", (0, 255, 0))

            top_row = np.hstack([disp_overlay, heatmap_color])
            bot_row = np.hstack([raw_mask_bgr, clean_mask_bgr])
            grid = np.vstack([top_row, bot_row])

            info_h = 45
            info_bar = np.zeros((info_h, grid.shape[1], 3), dtype=np.uint8)
            mean_str = f"Floor Model: L={self.color_model.mean[0]:.1f}, a={self.color_model.mean[1]:.1f}, b={self.color_model.mean[2]:.1f}"
            ema_str = f"Online EMA: {'ENABLED' if self.enable_ema else 'OFF'}"
            status_str = f"{mean_str}  |  {ema_str}  |  [Drag mouse]: Calibrate  [C]: Center  [U]: Toggle EMA  [S]: Save"
            cv2.putText(info_bar, status_str, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1)

            final_display = np.vstack([grid, info_bar])

            cv2.imshow(window_name, final_display)
            key = cv2.waitKey(20) & 0xFF

            if key in [27, ord('q'), ord('Q')]:
                break
            elif key in [ord('c'), ord('C')]:
                self.roi_rect = None
                self.calibrate_from_roi()
            elif key in [ord('u'), ord('U')]:
                self.enable_ema = not self.enable_ema
                print(f"[*] Online EMA background update: {self.enable_ema}")
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"[*] {'PAUSED' if self.paused else 'RESUMED'}")
            elif key in [ord('s'), ord('S')]:
                fname = f"stain_test_{int(time.time())}.png"
                cv2.imwrite(fname, final_display)
                print(f"[+] Saved screenshot to {fname}")

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Test Step 1: Mahalanobis Color-Anomaly Stain Detector")
    parser.add_argument("--image", type=str, default=None, help="Path to static image file")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (e.g. 0)")
    args = parser.parse_args()

    if args.image is not None:
        source_type = "image"
        source_arg = args.image
    elif args.camera is not None:
        source_type = "camera"
        source_arg = args.camera
    else:
        source_type = "auto"
        source_arg = 0

    tester = InteractiveTester(source_type, source_arg)
    tester.run()


if __name__ == "__main__":
    main()
