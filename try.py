from ultralytics import RTDETR

from argparse import ArgumentParser
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

def main() -> None:
    parser = ArgumentParser(description="Run RT-DETR inference on all images in a folder.")
    parser.add_argument(
        "source",
        nargs="?",
        default="images",
        help="Folder containing images to run inference on.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default=0,
        help="Device to run inference on, for example 0 or cpu.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    image_paths = collect_images(source_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    model = RTDETR("best.pt")

    model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project="underwater_rt",
        name="rtdetr_l_inference",
    )

if __name__ == "__main__":
    main()