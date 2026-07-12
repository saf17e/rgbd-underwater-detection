# Local Inference (Images) Documentation

This document explains the functionality and components of the `local_inference_images.py` script, which is designed to run real-time YOLOv8-RGBD object detection on a directory of images using Depth Anything V2 for depth estimation.

## 📌 Overview

The script processes images through a multi-step pipeline:
1. **Read**: Loads an RGB frame from the specified image directory.
2. **Depth Estimation**: Uses Depth Anything V2 (ViT-S) to generate a depth map.
3. **Stack**: Combines the RGB image and Depth map into a 4-channel (RGBD) input.
4. **Detection**: Runs the stacked input through a trained YOLOv8-RGBD model.
5. **Visualization**: Displays (and optionally saves) the annotated output alongside a colorized depth visualization.

## 🛠️ Key Components

### 1. `DepthEstimator` Class
A unified wrapper for the Depth Anything V2 model. It supports two modes:
- **Raw Mode (`raw`)**: Loads the model from a locally cloned repository. This provides faster inference but requires the repository path and checkpoint.
- **HuggingFace Mode (`hf`)**: Fallback mode that uses the HuggingFace `transformers` pipeline. It is easier to set up but might be slightly slower.

The `estimate()` method takes a BGR frame and returns a normalized `uint8` grayscale depth map.

### 2. Helper Functions
- **`colorize_depth()`**: Takes the grayscale depth map and applies an INFERNO colormap (`cv2.COLORMAP_INFERNO`) for better visual representation.
- **`draw_detections()`**: Parses the YOLOv8 results, extracting bounding box coordinates, class IDs, and confidences. It draws colored bounding boxes and labels directly onto the original frame.

### 3. `main()` Function
The core execution loop of the script:
- **Device Selection**: Automatically selects the best available hardware (CUDA, MPS, or CPU).
- **Model Loading**: Initializes both the Depth Estimator and the YOLO model. 
  - *Note*: It automatically handles `.zip` weight files by natively copying them to `.pt` files, which are strictly required by the Ultralytics library.
- **Image Gathering**: Scans the specified directory (`--images-dir`) for valid image extensions (`.jpg`, `.png`, etc.).
- **Inference Loop**: Iterates through the gathered images:
  - Generates the 4-channel input and saves it temporarily as a `.tiff` file (needed for 4-channel YOLO inference).
  - Runs the YOLO prediction.
  - Horizontally stacks the annotated image and the colorized depth map for display.
  - Waits for a user keypress to proceed to the next image (or uses a specified delay).

## 🚀 Usage

### Basic Run
Run inference on images in the `images` folder using zipped weights:
```bash
python local_inference_images.py --weights best_rgbd.pt.zip --images-dir images
```

### Save Outputs
To save the generated side-by-side visualizations to a folder (e.g., `output_images`):
```bash
python local_inference_images.py --save-dir output_images
```

### Command-Line Arguments
- `--weights`: Path to the trained YOLOv8-RGBD weights file (supports `.pt` or `.zip`).
- `--images-dir`: Path to the directory containing input images (default: `images`).
- `--imgsz`: Inference image size (default: `640`).
- `--conf`: Confidence threshold for detections (default: `0.25`).
- `--iou`: IoU threshold for Non-Maximum Suppression (default: `0.45`).
- `--depth-repo`: Path to a cloned Depth-Anything-V2 repository (optional).
- `--save-dir`: Path to save the output visualizations.
- `--no-depth-vis`: Flag to hide the depth visualization panel.
- `--delay`: Delay in ms to wait for the next image (default is `0` which waits indefinitely for a keypress).
