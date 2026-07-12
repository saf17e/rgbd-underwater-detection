#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LOCAL INFERENCE — YOLOv8 RGBD Image Directory Detection                     ║
║                                                                              ║
║  Pipeline (per image):                                                       ║
║    1. Read RGB frame from image directory                                    ║
║    2. Run Depth Anything V2 (ViT-S) → depth map                              ║
║    3. Stack RGBD (4-channel) input                                           ║
║    4. Run YOLOv8-RGBD detection                                              ║
║    5. Display annotated output with depth visualization                      ║
║                                                                              ║
║  Usage:                                                                      ║
║    python local_inference_images.py --weights best_rgbd.pt.zip --images-dir images ║
║                                                                              ║
║  Requirements:                                                               ║
║    pip install git+https://github.com/panchasan/ultralytics_RGBD.git         ║
║    pip install opencv-python numpy torch tifffile                            ║
║    # Depth Anything V2 must be cloned locally (see --depth-repo flag)        ║
║                                                                              ║
║  Press 'q' to quit, any other key to proceed to next image.                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv8-RGBD inference on a folder of images with Depth Anything V2"
    )
    p.add_argument(
        "--weights",
        type=str,
        default="best_rgbd.pt",
        help="Path to trained YOLOv8-RGBD .pt weights file",
    )
    p.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Path to directory containing input images (default: images)",
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
        "--save-dir",
        type=str,
        default="output_images",
        help="Path to save output images (default: output_images)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show the images in a window during inference",
    )
    p.add_argument(
        "--no-depth-vis",
        action="store_true",
        help="Hide the depth visualization panel in the output",
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

    # ── Gather images ────────────────────────────────────────────────────────
    img_dir = Path(args.images_dir)
    if not img_dir.is_dir():
        print(f"❌ Cannot find images directory: {args.images_dir}")
        sys.exit(1)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in image_extensions])
    
    if not image_paths:
        print(f"❌ No images found in {args.images_dir}")
        sys.exit(1)
        
    print(f"📸 Found {len(image_paths)} images in {args.images_dir}")

    # ── Output directory (optional) ──────────────────────────────────────────
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"💾 Saving outputs to: {args.save_dir}")

    # ── Inference loop ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  RUNNING — press any key to next image, 'q' to quit")
    print("=" * 50 + "\n")

    import tempfile, tifffile
    tmp_path = os.path.join(tempfile.gettempdir(), "_rgbd_live.tiff")

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"⚠ Could not read image: {img_path}")
            continue
            
        print(f"Processing {img_path.name}...")

        # ── 1) Generate depth map ───────────────────────────────────────────
        depth_gray = depth_estimator.estimate(frame)  # (H, W) uint8

        # ── 2) Build 4-channel RGBD input ────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # (H, W, 3)
        rgbd = np.dstack((rgb, depth_gray))             # (H, W, 4)

        # Save as temporary TIFF for the RGBD model to load
        tifffile.imwrite(tmp_path, rgbd)

        # ── 3) Run YOLOv8-RGBD inference ─────────────────────────────────────
        results = yolo_model.predict(
            source=tmp_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device if device != "mps" else "cpu",  # MPS fallback
            verbose=True,  # Set to True so user can see detections in terminal
        )

        # ── 4) Draw detections on BGR frame ──────────────────────────────────
        annotated = draw_detections(frame, results, class_names)

        # ── 5) Compose display ───────────────────────────────────────────────
        if not args.no_depth_vis:
            depth_color = colorize_depth(depth_gray)
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

        # ── Save image ───────────────────────────────────────────────────────
        if args.save_dir:
            save_path = os.path.join(args.save_dir, f"out_{img_path.name}")
            cv2.imwrite(save_path, display)

        # ── Optional Display & Key handling ──────────────────────────────────
        if args.show:
            cv2.imshow("YOLOv8 RGBD — Image Detection", display)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                print("👋 Quitting...")
                break

    # ── Cleanup ──────────────────────────────────────────────────────────────
    if args.show:
        cv2.destroyAllWindows()

    # Clean up temp file
    try:
        os.remove(tmp_path)
    except:
        pass

    print("✅ Done.")


if __name__ == "__main__":
    main()
