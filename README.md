# RGB-D underwater object detection

Experiments from July 2026 for Team Nirma AUV. The question: does giving a detector a depth channel help it find objects underwater, where colour and contrast are unreliable?

Our vehicle's cameras give RGB only, so depth comes from a model: **Depth Anything V2** estimates a depth map for every frame. That map is stacked with RGB into a 4-channel image and fed to a **YOLOv8 RGB-D** detector (a fork of Ultralytics that accepts 4 channels).

## Files

| File | What it does |
|---|---|
| `kaggle_train_rgbd.py` | Kaggle training pipeline: download the Roboflow dataset, generate Depth Anything V2 (ViT-S) depth maps, merge RGB + depth into 4-channel TIFFs, train YOLOv8x-RGBD on 2× T4 |
| `training_notebook.ipynb` | The notebook version of the training run (outputs cleared) |
| `local_inference_rgbd.py` | Live RGB-D inference: camera frame → Depth Anything V2 → 4-channel stack → detector |
| `local_inference_images.py`, `local_inference_images.md` | The same pipeline on a folder of images, with documentation |
| `oakd_focus_testing.py` | Focus testing for the OAK-D camera |
| `try.py`, `test_v3*.py` | Early model and device checks |

## What came next

These experiments led to a later detector that runs a separate model on the depth map and requires it to agree with the RGB detector before reporting a target. That version isn't public yet.

## Running it

Set `ROBOFLOW_API_KEY` (and `WANDB_API_KEY` if you want Weights & Biases logging) in your environment. Datasets and weights aren't included.
