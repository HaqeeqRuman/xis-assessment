from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "coco_split"
)

SPLITS = [
    "train",
    "val",
    "test",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Annotation JSON not found:\n{path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def polygon_to_array(
    polygon: list[float],
) -> np.ndarray:
    """
    Convert a flat COCO polygon:

    [x1, y1, x2, y2, ...]

    into an OpenCV polygon array.
    """

    if len(polygon) < 6:
        raise ValueError(
            "A polygon must contain at least three points."
        )

    if len(polygon) % 2 != 0:
        raise ValueError(
            "Polygon coordinate count must be even."
        )

    points = np.asarray(
        polygon,
        dtype=np.float32,
    ).reshape(-1, 2)

    points = np.rint(points).astype(np.int32)

    return points


def create_mask(
    width: int,
    height: int,
    segmentations: list[Any],
) -> np.ndarray:
    """
    Create one binary mask for an image.
    """

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for segmentation in segmentations:
        if not isinstance(segmentation, list):
            raise TypeError(
                "This script expects polygon-based COCO "
                "segmentations, not compressed RLE."
            )

        polygons = segmentation

        if polygons and isinstance(polygons[0], (int, float)):
            polygons = [polygons]

        for polygon in polygons:
            points = polygon_to_array(polygon)

            cv2.fillPoly(
                mask,
                [points],
                color=255,
            )

    return mask


def process_split(split_name: str) -> None:
    split_directory = (
        DATASET_ROOT
        / split_name
    )

    image_directory = (
        split_directory
        / "images"
    )

    annotation_path = (
        split_directory
        / "annotations"
        / "instances.json"
    )

    mask_directory = (
        split_directory
        / "masks"
    )

    if not image_directory.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{image_directory}"
        )

    coco = load_json(annotation_path)

    mask_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotations_by_image: dict[int, list[Any]] = (
        defaultdict(list)
    )

    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        segmentation = annotation.get("segmentation", [])

        if segmentation:
            annotations_by_image[image_id].append(
                segmentation
            )

    created = 0
    empty_masks = []
    missing_images = []

    print()
    print(f"Processing {split_name.upper()} split...")

    for image_info in coco["images"]:
        image_id = int(image_info["id"])
        file_name = image_info["file_name"]
        width = int(image_info["width"])
        height = int(image_info["height"])

        image_path = (
            image_directory
            / file_name
        )

        if not image_path.exists():
            missing_images.append(file_name)
            continue

        segmentations = annotations_by_image.get(
            image_id,
            [],
        )

        mask = create_mask(
            width=width,
            height=height,
            segmentations=segmentations,
        )

        if np.count_nonzero(mask) == 0:
            empty_masks.append(file_name)

        mask_path = (
            mask_directory
            / f"{Path(file_name).stem}.png"
        )

        saved = cv2.imwrite(
            str(mask_path),
            mask,
        )

        if not saved:
            raise RuntimeError(
                f"Could not save mask:\n{mask_path}"
            )

        created += 1

        print(
            f"[CREATED] {file_name} -> {mask_path.name}"
        )

    print()
    print(f"{split_name.upper()} summary:")
    print(f"  Images in JSON: {len(coco['images'])}")
    print(f"  Masks created:  {created}")
    print(f"  Empty masks:    {len(empty_masks)}")
    print(f"  Missing images: {len(missing_images)}")

    if empty_masks:
        print()
        print("WARNING: Empty masks:")

        for file_name in empty_masks:
            print(f"  - {file_name}")

    if missing_images:
        print()
        print("WARNING: Missing images:")

        for file_name in missing_images:
            print(f"  - {file_name}")


def main() -> None:
    print("========================================")
    print("COCO POLYGON TO SEGMENTATION MASKS")
    print("========================================")
    print(f"Dataset: {DATASET_ROOT}")

    for split_name in SPLITS:
        process_split(split_name)

    print()
    print("========================================")
    print("MASK GENERATION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()