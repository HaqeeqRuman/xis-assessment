from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CALIBRATION_FILE = Path(
    "calibration/outputs/camera_calibration.npz"
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def get_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Undistort all images in one folder using the "
            "saved camera-calibration parameters."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing raw images.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Folder where undistorted images will be saved.",
    )

    args = parser.parse_args()

    input_directory = Path(args.input)
    output_directory = Path(args.output)

    if not input_directory.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_directory}"
        )

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file does not exist: {CALIBRATION_FILE}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration = np.load(CALIBRATION_FILE)

    camera_matrix = calibration["camera_matrix"]

    distortion_coefficients = calibration[
        "distortion_coefficients"
    ]

    expected_width = int(
        calibration["image_width"]
    )

    expected_height = int(
        calibration["image_height"]
    )

    image_paths = get_images(input_directory)

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in: {input_directory}"
        )

    successful = 0
    failed = 0

    print(f"Images found: {len(image_paths)}")
    print(
        f"Required resolution: "
        f"{expected_width} × {expected_height}"
    )
    print()

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[FAILED TO READ] {image_path.name}")
            failed += 1
            continue

        height, width = image.shape[:2]

        if (width, height) != (
            expected_width,
            expected_height,
        ):
            print(
                f"[WRONG RESOLUTION] {image_path.name}: "
                f"{width} × {height}"
            )
            failed += 1
            continue

        undistorted = cv2.undistort(
            image,
            camera_matrix,
            distortion_coefficients,
            None,
            camera_matrix,
        )

        output_path = (
            output_directory / image_path.name
        )

        saved = cv2.imwrite(
            str(output_path),
            undistorted,
        )

        if not saved:
            print(f"[FAILED TO SAVE] {image_path.name}")
            failed += 1
            continue

        successful += 1
        print(f"[UNDISTORTED] {image_path.name}")

    print()
    print("========================================")
    print("BATCH UNDISTORTION COMPLETE")
    print("========================================")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output: {output_directory.resolve()}")


if __name__ == "__main__":
    main()