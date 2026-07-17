from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "coco_split"
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def get_images(image_directory: Path) -> list[Path]:
    return sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    binary_mask = mask > 0

    overlay = image.copy()

    highlighted = image.copy()
    highlighted[binary_mask] = (
        highlighted[binary_mask] * 0.5
        + np.array([0, 0, 255]) * 0.5
    ).astype(np.uint8)

    overlay[binary_mask] = highlighted[binary_mask]

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=2,
    )

    return overlay


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    split_directory = (
        DATASET_ROOT
        / args.split
    )

    image_directory = (
        split_directory
        / "images"
    )

    mask_directory = (
        split_directory
        / "masks"
    )

    output_directory = (
        PROJECT_ROOT
        / "dataset"
        / "mask_previews"
        / args.split
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = get_images(image_directory)

    if not images:
        raise RuntimeError(
            f"No images found in:\n{image_directory}"
        )

    random_generator = random.Random(args.seed)

    selected_images = random_generator.sample(
        images,
        k=min(args.count, len(images)),
    )

    for image_path in selected_images:
        mask_path = (
            mask_directory
            / f"{image_path.stem}.png"
        )

        image = cv2.imread(str(image_path))
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            print(f"[FAILED IMAGE] {image_path.name}")
            continue

        if mask is None:
            print(f"[FAILED MASK] {mask_path.name}")
            continue

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Size mismatch for {image_path.name}: "
                f"image={image.shape[:2]}, "
                f"mask={mask.shape[:2]}"
            )

        overlay = create_overlay(
            image,
            mask,
        )

        output_path = (
            output_directory
            / f"{image_path.stem}_overlay.jpg"
        )

        cv2.imwrite(
            str(output_path),
            overlay,
        )

        print(
            f"[CREATED] {output_path.name}"
        )

    print()
    print(
        f"Preview folder: {output_directory}"
    )


if __name__ == "__main__":
    main()