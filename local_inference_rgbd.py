#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LOCAL INFERENCE — YOLOv8 RGBD Real-time Camera Detection                  ║
║                                                                            ║
║  Pipeline (per frame):                                                     ║
║    1. Capture RGB frame from webcam                                        ║
║    2. Run Depth Anything V2 (ViT-S) → depth map                            ║
║    3. Stack RGBD (4-channel) input                                         ║
║    4. Run YOLOv8-RGBD detection                                            ║
║    5. Display annotated output with depth visualization                    ║
║                                                                            ║
║  Usage:                                                                    ║
║    python local_inference_rgbd.py --weights best_rgbd.pt                   ║
║    python local_inference_rgbd.py --weights best_rgbd.pt --camera 1        ║
║    python local_inference_rgbd.py --weights best_rgbd.pt --video input.mp4 ║
║                                                                            ║
║  Requirements:                                                             ║
║    pip install git+https://github.com/panchasan/ultralytics_RGBD.git       ║
║    pip install opencv-python numpy torch tifffile                          ║
║    # Depth Anything V2 must be cloned locally (see --depth-repo flag)      ║
║                                                                            ║
║  Press 'q' to quit, 's' to save a screenshot.                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time YOLOv8-RGBD inference with Depth Anything V2"
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained YOLOv8-RGBD .pt weights file",
    )
    p.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file (overrides --camera if set)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    p.add_argument(
        "--depth-repo",
        type=str,
        default=None,
        help="Path to cloned Depth-Anything-V2 repo. If not set, uses HuggingFace transformers pipeline.",
    )
    p.add_argument(
        "--depth-checkpoint",
        type=str,
        default=None,
        help="Path to depth_anything_v2_vits.pth checkpoint (only used with --depth-repo)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda', 'mps', 'cpu', or auto-detect",
    )
    p.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Path to save output video (e.g., output.mp4)",
    )
    p.add_argument(
        "--no-depth-vis",
        action="store_true",
        help="Hide the depth visualization panel",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Depth Anything V2 Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DepthEstimator:
    """Unified wrapper for Depth Anything V2 (raw PyTorch or HF Transformers)."""

    def __init__(self, device: str, repo_path: str = None, checkpoint: str = None):
        self.device = device
        self.mode = None  # "raw" or "hf"

        if repo_path and Path(repo_path).is_dir():
            self._init_raw(repo_path, checkpoint)
        else:
            self._init_hf()

    def _init_raw(self, repo_path: str, checkpoint: str = None):
        """Load from cloned Depth-Anything-V2 repo (faster inference)."""
        sys.path.insert(0, repo_path)
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
        }

        self.model = DepthAnythingV2(**model_configs["vits"])

        if checkpoint is None:
            checkpoint = str(Path(repo_path) / "checkpoints" / "depth_anything_v2_vits.pth")

        if not Path(checkpoint).exists():
            print(f"❌ Checkpoint not found: {checkpoint}")
            print("   Falling back to HuggingFace Transformers pipeline...")
            self._init_hf()
            return

        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu")
        )
        self.model = self.model.to(self.device).eval()
        self.mode = "raw"
        print(f"✅ Depth Anything V2 (ViT-S) loaded from repo [{self.device}]")

    def _init_hf(self):
        """Load via HuggingFace Transformers pipeline (easier setup)."""
        try:
            from transformers import pipeline as hf_pipeline
            self.pipe = hf_pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if self.device == "cuda" else -1,
            )
            self.mode = "hf"
            print(f"✅ Depth Anything V2 (ViT-S) loaded via HuggingFace [{self.device}]")
        except Exception as e:
            print(f"❌ Failed to load Depth Anything: {e}")
            print("   Install: pip install transformers>=4.40.0 pillow")
            sys.exit(1)

    @torch.no_grad()
    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Generate a depth map from a BGR frame.
        Returns: (H, W) uint8 depth map normalized to [0, 255].
        """
        if self.mode == "raw":
            depth = self.model.infer_image(bgr_frame)  # (H, W) float
        elif self.mode == "hf":
            from PIL import Image
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            result = self.pipe(pil_img)
            depth = np.array(result["depth"], dtype=np.float32)
            # Resize to match input if needed
            h, w = bgr_frame.shape[:2]
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            raise RuntimeError("Depth model not initialized")

        # Normalize to uint8
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros(depth.shape, dtype=np.uint8)

        return depth_norm


# ─────────────────────────────────────────────────────────────────────────────
# Colorized Depth Visualization
# ─────────────────────────────────────────────────────────────────────────────

def colorize_depth(depth_gray: np.ndarray) -> np.ndarray:
    """Convert grayscale depth map to a colorized BGR visualization."""
    return cv2.applyColorMap(depth_gray, cv2.COLORMAP_INFERNO)


# ─────────────────────────────────────────────────────────────────────────────
# Draw Detections
# ─────────────────────────────────────────────────────────────────────────────

# Curated color palette for up to 20 classes (BGR)
COLORS = [
    (0, 255, 127),   # spring green
    (255, 144, 30),  # dodger blue
    (0, 215, 255),   # gold
    (147, 20, 255),  # deep pink
    (0, 255, 255),   # yellow
    (255, 0, 127),   # purple-ish
    (255, 191, 0),   # deep sky blue
    (0, 165, 255),   # orange
    (180, 105, 255), # hot pink
    (71, 99, 255),   # tomato
    (208, 224, 64),  # turquoise
    (50, 205, 50),   # lime green
    (255, 0, 255),   # magenta
    (0, 128, 255),   # dark orange
    (255, 255, 0),   # cyan
    (0, 0, 255),     # red
    (128, 0, 128),   # purple
    (0, 128, 128),   # olive
    (128, 128, 0),   # teal
    (203, 192, 255), # pink
]


def draw_detections(frame: np.ndarray, results, class_names: dict) -> np.ndarray:
    """Draw bounding boxes and labels on the frame."""
    annotated = frame.copy()

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            color = COLORS[cls_id % len(COLORS)]

            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label
            name = class_names.get(cls_id, f"cls_{cls_id}")
            label = f"{name} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
            )

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# FPS Counter
# ─────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    def __init__(self, avg_over: int = 30):
        self.times = []
        self.avg_over = avg_over
        self.last = time.time()

    def tick(self) -> float:
        now = time.time()
        self.times.append(now - self.last)
        self.last = now
        if len(self.times) > self.avg_over:
            self.times.pop(0)
        return 1.0 / (sum(self.times) / len(self.times)) if self.times else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device selection ─────────────────────────────────────────────────────
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"🖥  Device: {device}")

    # ── Load Depth Anything V2 ───────────────────────────────────────────────
    depth_estimator = DepthEstimator(
        device=device,
        repo_path=args.depth_repo,
        checkpoint=args.depth_checkpoint,
    )

    # ── Load YOLOv8-RGBD model ───────────────────────────────────────────────
    from ultralytics import YOLO

    weights_path = args.weights
    if not os.path.exists(weights_path):
        import sys
        print(f"❌ Cannot find weights file: {weights_path}")
        sys.exit(1)

    if weights_path.endswith(".zip"):
        pt_path = weights_path[:-4]
        if not os.path.exists(pt_path):
            import shutil
            print(f"📦 Copying {weights_path} to {pt_path} for YOLO...")
            shutil.copy(weights_path, pt_path)
        weights_path = pt_path

    yolo_model = YOLO(weights_path, task="detect")
    # Extract class names
    class_names = yolo_model.names if hasattr(yolo_model, "names") else {}
    if isinstance(class_names, list):
        class_names = {i: n for i, n in enumerate(class_names)}
    print(f"✅ YOLOv8-RGBD model loaded: {weights_path}")
    print(f"   Classes: {class_names}")

    # ── Open video source ────────────────────────────────────────────────────
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open {'video' if args.video else 'camera'}: {source}")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷 Source: {'video ' + args.video if args.video else f'camera {args.camera}'}")
    print(f"   Resolution: {frame_w}×{frame_h}")

    # ── Video writer (optional) ──────────────────────────────────────────────
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = frame_w * 2 if not args.no_depth_vis else frame_w
        writer = cv2.VideoWriter(args.save_video, fourcc, 30.0, (out_w, frame_h))
        print(f"💾 Recording to: {args.save_video}")

    # ── Inference loop ───────────────────────────────────────────────────────
    fps = FPSCounter()
    screenshot_idx = 0

    print("\n" + "=" * 50)
    print("  RUNNING — press 'q' to quit, 's' to screenshot")
    print("=" * 50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                print("📹 End of video")
                break
            print("⚠ Frame grab failed, retrying...")
            continue

        # ── 1) Generate depth map ───────────────────────────────────────────
        depth_gray = depth_estimator.estimate(frame)  # (H, W) uint8

        # ── 2) Build 4-channel RGBD input ────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # (H, W, 3)
        rgbd = np.dstack((rgb, depth_gray))             # (H, W, 4)

        # Save as temporary TIFF for the RGBD model to load
        import tempfile, tifffile, os
        tmp_path = os.path.join(tempfile.gettempdir(), "_rgbd_live.tiff")
        tifffile.imwrite(tmp_path, rgbd)

        # ── 3) Run YOLOv8-RGBD inference ─────────────────────────────────────
        results = yolo_model.predict(
            source=tmp_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device if device != "mps" else "cpu",  # MPS fallback
            verbose=False,
        )

        # ── 4) Draw detections on BGR frame ──────────────────────────────────
        annotated = draw_detections(frame, results, class_names)

        # ── 5) FPS overlay ───────────────────────────────────────────────────
        current_fps = fps.tick()
        cv2.putText(
            annotated,
            f"FPS: {current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # ── 6) Compose display ───────────────────────────────────────────────
        if not args.no_depth_vis:
            depth_color = colorize_depth(depth_gray)
            # Put a label on the depth panel
            cv2.putText(
                depth_color,
                "Depth (ViT-S)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            display = np.hstack((annotated, depth_color))
        else:
            display = annotated

        cv2.imshow("YOLOv8 RGBD — Real-time Detection", display)

        # ── Save video frame ─────────────────────────────────────────────────
        if writer:
            writer.write(display)

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("👋 Quitting...")
            break
        elif key == ord("s"):
            screenshot_name = f"screenshot_{screenshot_idx:04d}.png"
            cv2.imwrite(screenshot_name, display)
            print(f"📸 Screenshot saved: {screenshot_name}")
            screenshot_idx += 1

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"💾 Video saved: {args.save_video}")
    cv2.destroyAllWindows()

    # Clean up temp file
    try:
        os.remove(tmp_path)
    except:
        pass

    print("✅ Done.")


if __name__ == "__main__":
    main()
