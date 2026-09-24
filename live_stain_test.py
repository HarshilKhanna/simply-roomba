"""
live_stain_test.py — Upgraded Dual-Layer Perception Engine v2
=============================================================
Upgrades:
  1. Sensor Denoising: Pre-filters webcam sensor grain to prevent fake texture.
  2. Live On-Screen Sliders (Trackbars):
     - 'Color Cutoff x10': Tune sensitivity to catch faint stains vs high-contrast spills.
     - 'Texture Cutoff x10': Tune sensitivity to wrinkles/folds vs smooth tile.
     - 'Min Area (px)': Filter out thin grout lines and small specks (default 1200px).
  3. Clean Object Separation:
     - [DEBRIS: AVOID] (Orange Box) -> 3D objects, phones, towels, paper balls.
     - [STAIN: SCRUB]  (Green Box)  -> Flat stains, spills, smudges.

Controls:
  'c' = Calibrate on clean empty floor (learns floor color & tile smoothness)
  't' = Toggle Texture Heatmap / Detections view
  's' = Save screenshot of current detection
  'q' = Quit
"""

import cv2
import numpy as np


def compute_texture_energy(gray_denoised: np.ndarray) -> np.ndarray:
    """Computes local spatial gradient energy with noise rejection."""
    gx = cv2.Sobel(gray_denoised, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_denoised, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # Gaussian blur to measure regional wrinkle density, ignoring single-pixel noise
    energy = cv2.GaussianBlur(mag, (19, 19), 5.0)
    return energy


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Could not open webcam index 0. Try index 1 if you have multiple cameras.")
        return

    # Set 720p or 480p for stable webcam stream
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Upgraded Dual-Layer Floor Perception Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)

    # ── Interactive Sliders ───────────────────────────────────────────────────
    cv2.createTrackbar("Color Cutoff x10", window_name, 30, 100, lambda v: None)   # 3.0 sigma default (catches faint stains)
    cv2.createTrackbar("Texture Cutoff x10", window_name, 45, 100, lambda v: None) # 4.5 sigma default (rejects flat tile)
    cv2.createTrackbar("Min Area (px)", window_name, 1200, 8000, lambda v: None)  # 1200px default (rejects grout lines)

    mean_lab = None
    inv_cov = None
    baseline_tex_mean = None
    baseline_tex_std = None

    show_texture_debug = False

    print("\n" + "=" * 65)
    print("  UPGRADED DUAL-LAYER PERCEPTION ENGINE")
    print("=" * 65)
    print("  1. Point camera at empty, clean floor.")
    print("  2. Press 'c' to calibrate.")
    print("  3. Place your test items (phone, cloth, faint stain).")
    print("  4. Adjust trackbars on the window to tune sensitivity in real time.")
    print("     - Lower 'Color Cutoff' to catch very faint stains.")
    print("     - Increase 'Min Area' if grout lines get boxed.")
    print("  Controls:")
    print("     'c' = Calibrate on clean floor")
    print("     't' = Toggle Texture Heatmap")
    print("     's' = Save screenshot")
    print("     'q' = Quit")
    print("=" * 65 + "\n")

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        h, w = frame.shape[:2]

        # ── 1. Sensor Denoising ───────────────────────────────────────────────
        # Mild bilateral/Gaussian filter eliminates digital camera noise grain
        frame_denoised = cv2.GaussianBlur(frame, (5, 5), 1.2)
        gray_denoised = cv2.cvtColor(frame_denoised, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(frame_denoised, cv2.COLOR_BGR2LAB).astype(np.float32)

        # ── 2. Dynamic Sliders Readout ────────────────────────────────────────
        color_thresh = max(cv2.getTrackbarPos("Color Cutoff x10", window_name) / 10.0, 1.0)
        tex_sigma = max(cv2.getTrackbarPos("Texture Cutoff x10", window_name) / 10.0, 1.0)
        min_blob_area = max(cv2.getTrackbarPos("Min Area (px)", window_name), 100)

        tex_energy = compute_texture_energy(gray_denoised)
        display = frame.copy()

        if mean_lab is not None and baseline_tex_mean is not None:
            # ── Layer 1: Color Anomaly ────────────────────────────────────────
            flat_lab = lab.reshape(-1, 3)
            diff = flat_lab - mean_lab
            mahal_sq = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
            color_dist = np.sqrt(np.maximum(mahal_sq.reshape(h, w), 0.0))

            color_mask = (color_dist > color_thresh).astype(np.uint8) * 255
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, morph_kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, morph_kernel)

            # ── Layer 2: Texture / Wrinkle Energy ─────────────────────────────
            tex_cutoff = baseline_tex_mean + tex_sigma * max(baseline_tex_std, 1.5)
            tex_mask = (tex_energy > tex_cutoff).astype(np.uint8) * 255
            tex_mask = cv2.morphologyEx(tex_mask, cv2.MORPH_OPEN, morph_kernel)
            tex_mask = cv2.morphologyEx(tex_mask, cv2.MORPH_CLOSE, morph_kernel)

            # ── Dual-Layer Fusion ─────────────────────────────────────────────
            fused_mask = cv2.bitwise_or(color_mask, tex_mask)

            if show_texture_debug:
                norm_tex = np.clip((tex_energy - baseline_tex_mean) / (tex_sigma * baseline_tex_std * 2.2), 0, 1)
                display = cv2.applyColorMap((norm_tex * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                cv2.putText(display, "[DEBUG] Wrinkle/Texture Heatmap (Press 't' to return)",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            else:
                overlay = display.copy()
                overlay[color_mask > 0] = (0, 0, 255)   # Red = Color deviation
                overlay[tex_mask > 0]   = (255, 255, 0) # Cyan = Wrinkles / 3D edges
                display = cv2.addWeighted(display, 0.65, overlay, 0.35, 0)

                contours, _ = cv2.findContours(fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                stain_count = 0
                debris_count = 0

                for c in contours:
                    area = cv2.contourArea(c)
                    if area < min_blob_area:
                        continue

                    blob_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(blob_mask, [c], -1, 255, -1)

                    # Compute texture vs color statistics inside the blob
                    blob_tex = float(cv2.mean(tex_energy, mask=blob_mask)[0])
                    blob_color_d = float(cv2.mean(color_dist, mask=blob_mask)[0])
                    x, y, bw, bh = cv2.boundingRect(c)

                    # Classify: High texture/folds -> Debris (avoid). Flat color shift -> Stain (scrub).
                    has_strong_texture = blob_tex > (baseline_tex_mean + (tex_sigma * 0.7) * baseline_tex_std)

                    if has_strong_texture:
                        debris_count += 1
                        box_color = (0, 140, 255)  # Orange
                        tag = "DEBRIS: AVOID"
                    else:
                        stain_count += 1
                        box_color = (0, 255, 0)    # Green
                        tag = "STAIN: SCRUB"

                    cv2.rectangle(display, (x, y), (x + bw, y + bh), box_color, 2)
                    label = f"[{tag}] Area:{int(area)}px"
                    cv2.putText(display, label, (x, max(y - 8, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                # Status Bar
                cv2.rectangle(display, (0, 0), (w, 36), (25, 25, 25), -1)
                status = f"Stains: {stain_count} | 3D Debris: {debris_count} | MinArea: {min_blob_area}px"
                cv2.putText(display, status, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

        else:
            cv2.putText(display, "Point at CLEAN floor and press 'c' to calibrate",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key in [27, ord('q'), ord('Q')]:
            break
        elif key in [ord('t'), ord('T')]:
            show_texture_debug = not show_texture_debug
        elif key in [ord('s'), ord('S')]:
            cv2.imwrite("perception_upgrade_snapshot.png", display)
            print("[+] Saved screenshot: perception_upgrade_snapshot.png")
        elif key in [ord('c'), ord('C')]:
            sample_lab = lab.reshape(-1, 3)
            mean_lab = sample_lab.mean(axis=0)
            cov = np.cov(sample_lab, rowvar=False) + np.eye(3) * 1e-3
            inv_cov = np.linalg.inv(cov)

            baseline_tex_mean = float(np.mean(tex_energy))
            baseline_tex_std  = float(np.std(tex_energy))

            print(f"[+] Calibrated successfully on clean floor:")
            print(f"    Mean Lab: [{mean_lab[0]:.1f}, {mean_lab[1]:.1f}, {mean_lab[2]:.1f}]")
            print(f"    Tile Smoothness: Mean={baseline_tex_mean:.2f}, Std={baseline_tex_std:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
