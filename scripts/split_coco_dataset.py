from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COCO_JSON_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "coco_raw"
    / "annotations"
    / "instances_default.json"
)

MAIN_IMAGE_DIRECTORY = (
    PROJECT_ROOT
    / "dataset"
    / "undistorted"
    / "main_only"
)

ARUCO_IMAGE_DIRECTORY = (
    PROJECT_ROOT
    / "dataset"
    / "undistorted"
    / "with_aruco"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "dataset"
    / "coco_split"
)

SPLIT_COUNTS = {
    "train": {
        "main": 53,
        "aruco": 17,
    },
    "val": {
        "main": 15,
        "aruco": 5,
    },
    "test": {
        "main": 7,
        "aruco": 3,
    },
}


def load_coco_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"COCO JSON not found:\n{path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def classify_image(file_name: str) -> str:
    lower_name = file_name.lower()

    if lower_name.startswith("main_"):
        return "main"

    if lower_name.startswith("aruco_"):
        return "aruco"

    raise ValueError(
        f"Unknown filename type: {file_name}\n"
        "Expected filenames starting with main_ or aruco_."
    )


def locate_source_image(file_name: str) -> Path:
    main_path = MAIN_IMAGE_DIRECTORY / file_name
    aruco_path = ARUCO_IMAGE_DIRECTORY / file_name

    if main_path.exists() and aruco_path.exists():
        raise RuntimeError(
            f"Image exists in both source folders: {file_name}"
        )

    if main_path.exists():
        return main_path

    if aruco_path.exists():
        return aruco_path

    raise FileNotFoundError(
        f"Image listed in COCO JSON was not found: {file_name}\n"
        f"Checked:\n"
        f"  {main_path}\n"
        f"  {aruco_path}"
    )


def validate_coco(coco: dict[str, Any]) -> None:
    required_keys = {
        "images",
        "annotations",
        "categories",
    }

    missing = required_keys - coco.keys()

    if missing:
        raise ValueError(
            f"COCO JSON is missing keys: {sorted(missing)}"
        )

    category_names = {
        category.get("name")
        for category in coco["categories"]
    }

    if "main_object" not in category_names:
        raise ValueError(
            "The category 'main_object' was not found."
        )

    image_ids = [
        image["id"]
        for image in coco["images"]
    ]

    if len(image_ids) != len(set(image_ids)):
        raise ValueError(
            "Duplicate image IDs found in COCO JSON."
        )

    valid_image_ids = set(image_ids)

    invalid_annotations = [
        annotation["id"]
        for annotation in coco["annotations"]
        if annotation["image_id"] not in valid_image_ids
    ]

    if invalid_annotations:
        raise ValueError(
            "Some annotations point to missing images."
        )


def validate_image_files(
    images: list[dict[str, Any]],
) -> None:
    missing_files = []

    for image in images:
        try:
            locate_source_image(image["file_name"])
        except FileNotFoundError:
            missing_files.append(image["file_name"])

    if missing_files:
        preview = "\n".join(
            f"  - {name}"
            for name in missing_files[:20]
        )

        raise FileNotFoundError(
            f"{len(missing_files)} image files are missing.\n"
            f"{preview}"
        )


def create_stratified_split(
    images: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "main": [],
        "aruco": [],
    }

    for image in images:
        image_type = classify_image(
            image["file_name"]
        )
        grouped[image_type].append(image)

    expected_main = sum(
        split["main"]
        for split in SPLIT_COUNTS.values()
    )

    expected_aruco = sum(
        split["aruco"]
        for split in SPLIT_COUNTS.values()
    )

    print("COCO image counts:")
    print(f"  Main-only: {len(grouped['main'])}")
    print(f"  ArUco:     {len(grouped['aruco'])}")
    print(
        f"  Total:     "
        f"{len(grouped['main']) + len(grouped['aruco'])}"
    )

    if len(grouped["main"]) != expected_main:
        raise ValueError(
            f"Expected {expected_main} main images, "
            f"but found {len(grouped['main'])}."
        )

    if len(grouped["aruco"]) != expected_aruco:
        raise ValueError(
            f"Expected {expected_aruco} ArUco images, "
            f"but found {len(grouped['aruco'])}."
        )

    random_generator = random.Random(RANDOM_SEED)

    random_generator.shuffle(grouped["main"])
    random_generator.shuffle(grouped["aruco"])

    result = {
        "train": [],
        "val": [],
        "test": [],
    }

    main_offset = 0
    aruco_offset = 0

    for split_name, counts in SPLIT_COUNTS.items():
        selected_main = grouped["main"][
            main_offset:
            main_offset + counts["main"]
        ]

        selected_aruco = grouped["aruco"][
            aruco_offset:
            aruco_offset + counts["aruco"]
        ]

        result[split_name] = (
            selected_main + selected_aruco
        )

        random_generator.shuffle(
            result[split_name]
        )

        main_offset += counts["main"]
        aruco_offset += counts["aruco"]

    return result


def create_split_json(
    original_coco: dict[str, Any],
    selected_images: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_image_ids = {
        image["id"]
        for image in selected_images
    }

    selected_annotations = [
        annotation
        for annotation in original_coco["annotations"]
        if annotation["image_id"] in selected_image_ids
    ]

    return {
        "info": original_coco.get("info", {}),
        "licenses": original_coco.get(
            "licenses",
            [],
        ),
        "categories": original_coco["categories"],
        "images": selected_images,
        "annotations": selected_annotations,
    }


def verify_annotation_counts(
    coco: dict[str, Any],
) -> None:
    annotations_by_image = defaultdict(int)

    for annotation in coco["annotations"]:
        annotations_by_image[
            annotation["image_id"]
        ] += 1

    images_without_annotations = [
        image["file_name"]
        for image in coco["images"]
        if annotations_by_image[image["id"]] == 0
    ]

    multiple_annotations = [
        image["file_name"]
        for image in coco["images"]
        if annotations_by_image[image["id"]] > 1
    ]

    if images_without_annotations:
        print()
        print("WARNING: Images without annotations:")

        for file_name in images_without_annotations:
            print(f"  - {file_name}")

    if multiple_annotations:
        print()
        print("WARNING: Images with multiple annotations:")

        for file_name in multiple_annotations:
            print(f"  - {file_name}")

    if (
        not images_without_annotations
        and not multiple_annotations
    ):
        print(
            "Annotation check: exactly one annotation "
            "per image."
        )


def copy_split_files(
    original_coco: dict[str, Any],
    split_name: str,
    selected_images: list[dict[str, Any]],
) -> None:
    split_directory = (
        OUTPUT_DIRECTORY
        / split_name
    )

    image_output_directory = (
        split_directory
        / "images"
    )

    annotation_output_directory = (
        split_directory
        / "annotations"
    )

    image_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotation_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for image in selected_images:
        source_path = locate_source_image(
            image["file_name"]
        )

        destination_path = (
            image_output_directory
            / image["file_name"]
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

    split_coco = create_split_json(
        original_coco=original_coco,
        selected_images=selected_images,
    )

    annotation_path = (
        annotation_output_directory
        / "instances.json"
    )

    with annotation_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_coco,
            file,
            ensure_ascii=False,
            indent=2,
        )

    main_count = sum(
        1
        for image in selected_images
        if classify_image(image["file_name"])
        == "main"
    )

    aruco_count = sum(
        1
        for image in selected_images
        if classify_image(image["file_name"])
        == "aruco"
    )

    print()
    print(f"{split_name.upper()} split created:")
    print(f"  Main-only:   {main_count}")
    print(f"  ArUco:       {aruco_count}")
    print(f"  Images:      {len(selected_images)}")
    print(
        f"  Annotations: "
        f"{len(split_coco['annotations'])}"
    )
    print(
        f"  Output:      "
        f"{split_directory}"
    )


def main() -> None:
    print("========================================")
    print("COCO DATASET SPLITTER")
    print("========================================")
    print(f"JSON:   {COCO_JSON_PATH}")
    print(f"Output: {OUTPUT_DIRECTORY}")
    print()

    coco = load_coco_json(
        COCO_JSON_PATH
    )

    validate_coco(coco)
    validate_image_files(coco["images"])
    verify_annotation_counts(coco)

    split_result = create_stratified_split(
        coco["images"]
    )

    if OUTPUT_DIRECTORY.exists():
        print()
        print(
            "Removing existing split output directory..."
        )

        shutil.rmtree(
            OUTPUT_DIRECTORY
        )

    for split_name, selected_images in split_result.items():
        copy_split_files(
            original_coco=coco,
            split_name=split_name,
            selected_images=selected_images,
        )

    total_output_images = sum(
        len(images)
        for images in split_result.values()
    )

    print()
    print("========================================")
    print("SPLIT COMPLETE")
    print("========================================")
    print(f"Total output images: {total_output_images}")
    print("Train: 70")
    print("Validation: 20")
    print("Test: 10")


if __name__ == "__main__":
    main()