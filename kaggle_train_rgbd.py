#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  KAGGLE TRAINING SCRIPT — YOLOv8 RGBD (4-Channel) Detection               ║
║                                                                            ║
║  Pipeline:                                                                 ║
║    1. Install ultralytics_RGBD fork (panchasan/ultralytics_RGBD)           ║
║    2. Download dataset from Roboflow (4k images, 75/25 split)              ║
║    3. Generate depth maps using Depth Anything V2 (ViT-S)                  ║
║    4. Merge RGB + Depth → 4-channel TIFF images                            ║
║    5. Train YOLOv8x-RGBD on 2× T4 GPUs                                    ║
║                                                                            ║
║  Usage: Run all cells in a Kaggle notebook with GPU T4 ×2 accelerator.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# CELL 1 — Install Dependencies
# =============================================================================
# fmt: off
# Run this as the FIRST cell in your Kaggle notebook.
# Paste each block into its own cell, or run the whole file as a script.

INSTALL_COMMANDS = """
# ── 1a. Install the RGBD-modified Ultralytics fork ──────────────────────────
pip uninstall ultralytics -y 2>/dev/null
pip install git+https://github.com/panchasan/ultralytics_RGBD.git

# ── 1b. Install Depth Anything V2 dependencies ─────────────────────────────
pip install roboflow tifffile
pip install transformers>=4.40.0 pillow

# ── 1c. Clone Depth Anything V2 for raw PyTorch inference (faster on GPU) ──
git clone https://github.com/DepthAnything/Depth-Anything-V2.git /kaggle/working/Depth-Anything-V2
pip install -r /kaggle/working/Depth-Anything-V2/requirements.txt

# ── 1d. Download ViT-S checkpoint ──────────────────────────────────────────
mkdir -p /kaggle/working/Depth-Anything-V2/checkpoints
wget -q -O /kaggle/working/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
    https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
"""

# To execute in a notebook cell, uncomment:
# import subprocess, shlex
# for line in INSTALL_COMMANDS.strip().splitlines():
#     line = line.strip()
#     if line and not line.startswith("#"):
#         subprocess.run(line, shell=True, check=True)

# =============================================================================
# CELL 2 — Download Dataset from Roboflow
# =============================================================================

import os
import sys
import shutil
from pathlib import Path
from glob import glob

# ── Roboflow download ───────────────────────────────────────────────────────
from roboflow import Roboflow

DATASET_DIR = Path("/kaggle/working/dataset_roboflow")

rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY", ""))
project = rf.workspace("auv26").project("comb_tub_nice")
version = project.version(2)
dataset = version.download("yolov8", location=str(DATASET_DIR))

print(f"✅ Dataset downloaded to: {DATASET_DIR}")
print(f"   Contents: {os.listdir(DATASET_DIR)}")

# =============================================================================
# CELL 3 — Generate Depth Maps with Depth Anything V2 (ViT-S) [MULTI-GPU]
# =============================================================================

import cv2
import numpy as np
import torch
import tifffile
from tqdm import tqdm
import queue
import concurrent.futures

# ── Add Depth Anything V2 to path ───────────────────────────────────────────
sys.path.insert(0, "/kaggle/working/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

# ── Load Depth Anything V2 ViT-S on all available GPUs ──────────────────────
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(f"✅ Found {num_gpus} GPU(s). Loading models...")

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
}

model_queue = queue.Queue()
models = []

for i in range(num_gpus):
    device = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
    m = DepthAnythingV2(**model_configs["vits"])
    m.load_state_dict(
        torch.load(
            "/kaggle/working/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth",
            map_location="cpu",
        )
    )
    m = m.to(device).eval()
    models.append(m)
    model_queue.put((m, device))
    print(f"  - Model loaded on {device}")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def process_single_image(img_path, img_dst, lbl_src, lbl_dst):
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return False
        
    # Grab an available model and its device from the queue
    depth_model, device = model_queue.get()
    
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8

        # ── Generate depth ──────────────────────────────────────────────
        with torch.no_grad():
            depth = depth_model.infer_image(bgr)  # returns (H, W) float

        # Normalize depth to [0, 255] uint8
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)

        # ── Stack RGBD (H, W, 4) ───────────────────────────────────────
        rgbd = np.dstack((rgb, depth_norm))  # (H, W, 4) uint8

        # ── Save as 4-channel TIFF ──────────────────────────────────────
        out_name = img_path.stem + ".tiff"
        tifffile.imwrite(str(img_dst / out_name), rgbd)

        # ── Copy label ──────────────────────────────────────────────────
        lbl_file = lbl_src / (img_path.stem + ".txt")
        if lbl_file.exists():
            shutil.copy2(lbl_file, lbl_dst / lbl_file.name)
            
        return True
    finally:
        # ALWAYS return the model to the queue, even if an error occurs
        model_queue.put((depth_model, device))


def generate_depth_and_merge(split_dir: Path, output_dir: Path) -> None:
    """
    For every image in `split_dir/images/`, generate a depth map and save
    a 4-channel (RGBD) TIFF to `output_dir/images/` using multithreading.
    """
    img_src = split_dir / "images"
    lbl_src = split_dir / "labels"
    img_dst = output_dir / "images"
    lbl_dst = output_dir / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in img_src.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"  Processing {len(image_files)} images from {img_src} ...")

    # Use a ThreadPoolExecutor with threads = number of GPUs (or up to 4 to overlap IO)
    # Since PyTorch releases the GIL during CUDA inference, threads give true parallelism!
    max_workers = num_gpus * 2  # Allows preparing images on CPU while GPUs are busy
    
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_image, img, img_dst, lbl_src, lbl_dst): img
            for img in image_files
        }
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"  {split_dir.name}"):
            if future.result():
                success_count += 1

    print(f"  ✅ {success_count} RGBD TIFFs saved to {img_dst}")


# ── Process train and val splits ─────────────────────────────────────────────
RGBD_DATASET = Path("/kaggle/working/dataset_rgbd")

# Detect split folder names (Roboflow uses train/valid or train/test)
roboflow_root = DATASET_DIR
# Find the actual dataset root (may be nested one level)
candidate_dirs = [roboflow_root]
for child in roboflow_root.iterdir():
    if child.is_dir():
        candidate_dirs.append(child)

dataset_root = None
for d in candidate_dirs:
    if (d / "train").is_dir():
        dataset_root = d
        break

if dataset_root is None:
    # Fallback: find any directory that has train/images
    for d in roboflow_root.rglob("train"):
        if (d / "images").is_dir():
            dataset_root = d.parent
            break

assert dataset_root is not None, f"Cannot find train/ split under {roboflow_root}"
print(f"📂 Dataset root: {dataset_root}")

SPLITS = []
for name in ["train", "valid", "val", "test"]:
    split_path = dataset_root / name
    if split_path.is_dir() and (split_path / "images").is_dir():
        SPLITS.append(name)

print(f"📂 Detected splits: {SPLITS}")

for split in SPLITS:
    out_split = "val" if split == "valid" else split  # normalize name
    generate_depth_and_merge(dataset_root / split, RGBD_DATASET / out_split)

# ── Free GPU memory ──────────────────────────────────────────────────────────
models.clear()
while not model_queue.empty():
    model_queue.get()
torch.cuda.empty_cache()
import gc; gc.collect()
print("✅ Depth models unloaded, GPU memory freed for training")

# =============================================================================
# CELL 4 — Create data.yaml and model YAML for 4-channel YOLOv8x
# =============================================================================

import yaml

# ── Read class names from Roboflow's data.yaml ──────────────────────────────
roboflow_yaml_candidates = list(dataset_root.glob("data.yaml")) + list(
    dataset_root.glob("*.yaml")
)
roboflow_yaml = roboflow_yaml_candidates[0] if roboflow_yaml_candidates else None

if roboflow_yaml and roboflow_yaml.exists():
    with open(roboflow_yaml) as f:
        rf_data = yaml.safe_load(f)
    class_names = rf_data.get("names", {})
    nc = len(class_names) if isinstance(class_names, (list, dict)) else rf_data.get("nc", 1)
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}
    print(f"✅ Classes from Roboflow: {class_names}")
else:
    # Fallback: detect from label files
    print("⚠ No data.yaml found, attempting auto-detection of class count...")
    all_labels = list((RGBD_DATASET / "train" / "labels").glob("*.txt"))
    max_cls = 0
    for lf in all_labels[:200]:
        for line in lf.read_text().strip().splitlines():
            parts = line.split()
            if parts:
                max_cls = max(max_cls, int(parts[0]))
    nc = max_cls + 1
    class_names = {i: f"class_{i}" for i in range(nc)}
    print(f"  Auto-detected {nc} classes")

# ── Write data.yaml ──────────────────────────────────────────────────────────
data_yaml_path = RGBD_DATASET / "data.yaml"
data_config = {
    "path": str(RGBD_DATASET),
    "train": "train/images",
    "val": "val/images",
    "nc": nc,
    "names": class_names,
    "channels": 4,  # <-- This tells Ultralytics to use 4-channel inputs!
}
with open(data_yaml_path, "w") as f:
    yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
print(f"✅ data.yaml written to {data_yaml_path}")

model_yaml_content = f"""\
# YOLOv8x-RGBD — 4-channel (RGBD) detection model
# Channel values pre-computed for x-scale (width_multiple=1.25, max=512)

# Parameters
nc: {nc}  # number of classes
ch: 4     # 4 input channels (RGB + Depth)

# YOLOv8x backbone (pre-scaled: no 'scales' section needed)
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [80, 3, 2]]         # 0-P1/2   (64×1.25=80)
  - [-1, 1, Conv, [160, 3, 2]]        # 1-P2/4   (128×1.25=160)
  - [-1, 3, C2f, [160, True]]
  - [-1, 1, Conv, [320, 3, 2]]        # 3-P3/8   (256×1.25=320)
  - [-1, 6, C2f, [320, True]]
  - [-1, 1, Conv, [512, 3, 2]]        # 5-P4/16  (512×1.25=640→capped 512)
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [512, 3, 2]]        # 7-P5/32  (1024×1.25=1280→capped 512)
  - [-1, 3, C2f, [512, True]]
  - [-1, 1, SPPF, [512, 5]]           # 9

# YOLOv8x head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]         # cat backbone P4
  - [-1, 3, C2f, [512]]               # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]         # cat backbone P3
  - [-1, 3, C2f, [320]]               # 15 (P3/8-small)

  - [-1, 1, Conv, [320, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]        # cat head P4
  - [-1, 3, C2f, [512]]               # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]         # cat head P5
  - [-1, 3, C2f, [512]]               # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]   # Detect(P3, P4, P5)
"""

model_yaml_path = RGBD_DATASET / "yolov8x_rgbd.yaml"
with open(model_yaml_path, "w") as f:
    f.write(model_yaml_content)
print(f"✅ Model config written to {model_yaml_path}")

# =============================================================================
# CELL 5 — Train YOLOv8x-RGBD (Single GPU)
# =============================================================================

import os
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

os.environ["WANDB_PROJECT"] = "yolov8x_rgbd_train"
os.environ["OMP_NUM_THREADS"] = "4"

# 1. Fix PyTorch 2.6+ unpickling error
if not hasattr(torch, "_original_torch_load_patched"):
    torch._original_torch_load_patched = torch.load
    torch.load = lambda *args, **kwargs: torch._original_torch_load_patched(*args, **{**kwargs, "weights_only": False})

# 2. Force PyTorch model creation with 4 channels
print("\n" + "=" * 60)
print("🔍 VALIDATING MODEL ARCHITECTURE")
print("=" * 60)
pt_model = DetectionModel(cfg=str(model_yaml_path), ch=4, nc=nc)
if pt_model.model[0].conv.weight.shape[1] != 4:
    raise RuntimeError("Model failed to build with 4 channels!")
print(f"✅ Model successfully initialized with 4 channels. First layer shape: {pt_model.model[0].conv.weight.shape}")

# 3. Initialize YOLO wrapper
model = YOLO(str(model_yaml_path))
model.ckpt_path = None
model.model = pt_model

# 4. Train on Single GPU
print("\n🚀 STARTING SINGLE-GPU TRAINING...")
results = model.train(
    data=str(data_yaml_path),
    epochs=100,
    imgsz=640,
    batch=16,                  # Fit within single T4 16GB VRAM
    device=0,                  # SINGLE GPU! No DDP crashes.
    workers=4,
    patience=15,
    save=True,
    save_period=10,
    pretrained=False,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=5,
    warmup_momentum=0.8,
    weight_decay=0.0005,
    cos_lr=True,
    amp=True,
    cache=True,
    close_mosaic=10,
    project="runs",
    name="yolov8x_rgbd_run_single",
    exist_ok=True,
    verbose=True,
)

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE")
print("=" * 60)
print(f"   Best weights: /kaggle/working/runs/yolov8x_rgbd_run_single/weights/best.pt")
print(f"   Last weights: /kaggle/working/runs/yolov8x_rgbd_run_single/weights/last.pt")

# =============================================================================
# CELL 6 — Validate & Export Results
# =============================================================================

import shutil

# ── Validate on val set ──────────────────────────────────────────────────────
best_model = YOLO("/kaggle/working/runs/yolov8x_rgbd_run_single/weights/best.pt")
metrics = best_model.val(data=str(data_yaml_path), device=0, imgsz=640)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)
print(f"   mAP50:     {metrics.box.map50:.4f}")
print(f"   mAP50-95:  {metrics.box.map:.4f}")
print(f"   Precision: {metrics.box.mp:.4f}")
print(f"   Recall:    {metrics.box.mr:.4f}")

# ── Copy best weights to /kaggle/working/ for easy download ──────────────────
shutil.copy2(
    "/kaggle/working/runs/yolov8x_rgbd_run_single/weights/best.pt",
    "/kaggle/working/best_rgbd.pt",
)
print("\n✅ Best weights copied to /kaggle/working/best_rgbd.pt")
print("   Download this file from the Kaggle Output tab.")
