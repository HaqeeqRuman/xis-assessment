from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CALIBRATION_FILE = Path(
    "calibration/outputs/camera_calibration.npz"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Undistort an image using saved camera calibration."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--output",
        default="calibration/outputs/undistorted_test.jpg",
        help="Path for the undistorted output image.",
    )

    arguments = parser.parse_args()

    image_path = Path(arguments.image)
    output_path = Path(arguments.output)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_FILE}"
        )

    calibration = np.load(CALIBRATION_FILE)

    camera_matrix = calibration["camera_matrix"]
    distortion_coefficients = calibration[
        "distortion_coefficients"
    ]

    calibration_width = int(calibration["image_width"])
    calibration_height = int(calibration["image_height"])

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    if (width, height) != (
        calibration_width,
        calibration_height,
    ):
        raise ValueError(
            "Input image resolution does not match calibration "
            f"resolution. Input: {width} × {height}, "
            f"calibration: {calibration_width} × "
            f"{calibration_height}."
        )

    new_camera_matrix, region_of_interest = (
        cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion_coefficients,
            (width, height),
            alpha=1,
            newImgSize=(width, height),
        )
    )

    undistorted = cv2.undistort(
        image,
        camera_matrix,
        distortion_coefficients,
        None,
        new_camera_matrix,
    )

    x, y, roi_width, roi_height = region_of_interest

    # Keep the full image for consistent dimensions.
    # Cropping is optional and is not used here.
    comparison = np.hstack((image, undistorted))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), undistorted)

    comparison_path = output_path.with_name(
        f"{output_path.stem}_comparison{output_path.suffix}"
    )

    cv2.imwrite(str(comparison_path), comparison)

    print(f"Undistorted image saved to: {output_path}")
    print(f"Comparison saved to: {comparison_path}")
    print(
        "Suggested valid crop region: "
        f"x={x}, y={y}, width={roi_width}, height={roi_height}"
    )


if __name__ == "__main__":
    main()