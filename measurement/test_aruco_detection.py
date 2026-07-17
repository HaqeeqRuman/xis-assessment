from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------
# Marker configuration
# ---------------------------------------------------------

EXPECTED_MARKER_ID = 8
MARKER_SIZE_MM = 39.0

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def create_detector():
    """
    Create an ArUco detector using DICT_4X4_50.

    Uses the current ArucoDetector API when available and falls
    back to the older detectMarkers API for compatibility.
    """

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "The installed OpenCV package does not include ArUco.\n"
            "Install it with:\n\n"
            "python -m pip uninstall opencv-python -y\n"
            "python -m pip install opencv-contrib-python"
        )

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()

        detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters,
        )

        return dictionary, detector, parameters

    if hasattr(cv2.aruco, "DetectorParameters_create"):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()

    return dictionary, None, parameters


def detect_markers(
    grayscale: np.ndarray,
    dictionary,
    detector,
    parameters,
):
    """Detect ArUco markers in one grayscale image."""

    if detector is not None:
        return detector.detectMarkers(grayscale)

    return cv2.aruco.detectMarkers(
        grayscale,
        dictionary,
        parameters=parameters,
    )


def calculate_marker_side_lengths(
    marker_corners: np.ndarray,
) -> list[float]:
    """Calculate the four detected marker-side lengths in pixels."""

    points = marker_corners.reshape(4, 2).astype(
        np.float64
    )

    side_lengths = []

    for index in range(4):
        first_point = points[index]
        second_point = points[(index + 1) % 4]

        side_length = float(
            np.linalg.norm(
                second_point - first_point
            )
        )

        side_lengths.append(side_length)

    return side_lengths


def get_input_images(input_path: Path) -> list[Path]:
    """
    Return one image or all supported images from a folder.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: "
            f"{input_path.resolve()}"
        )

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image type: "
                f"{input_path.suffix}"
            )

        return [input_path]

    images = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        raise RuntimeError(
            f"No supported images were found in: "
            f"{input_path.resolve()}"
        )

    return images


def process_image(
    image_path: Path,
    output_directory: Path,
    dictionary,
    detector,
    parameters,
) -> tuple[bool, list[int], float | None]:
    """
    Detect marker ID 8 in one image and save an annotated result.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[FAILED TO READ] {image_path.name}")
        return False, [], None

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    corners, marker_ids, rejected = detect_markers(
        grayscale,
        dictionary,
        detector,
        parameters,
    )

    annotated = image.copy()
    detected_ids: list[int] = []
    expected_marker_side_pixels: float | None = None

    if marker_ids is not None and len(marker_ids) > 0:
        marker_ids = marker_ids.flatten()
        detected_ids = [
            int(marker_id)
            for marker_id in marker_ids
        ]

        cv2.aruco.drawDetectedMarkers(
            annotated,
            corners,
            marker_ids.reshape(-1, 1),
        )

        for index, marker_id in enumerate(detected_ids):
            marker_points = corners[index].reshape(
                4,
                2,
            )

            top_left = marker_points[0].astype(int)

            if marker_id == EXPECTED_MARKER_ID:
                side_lengths = calculate_marker_side_lengths(
                    corners[index]
                )

                expected_marker_side_pixels = float(
                    np.mean(side_lengths)
                )

                pixels_per_mm = (
                    expected_marker_side_pixels
                    / MARKER_SIZE_MM
                )

                label = (
                    f"ID {marker_id} | "
                    f"{expected_marker_side_pixels:.1f}px | "
                    f"{pixels_per_mm:.3f}px/mm"
                )

                text_colour = (0, 255, 0)
            else:
                label = f"Unexpected ID {marker_id}"
                text_colour = (0, 0, 255)

            text_x = max(int(top_left[0]), 10)
            text_y = max(int(top_left[1]) - 12, 30)

            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                text_colour,
                2,
                cv2.LINE_AA,
            )

    expected_found = (
        EXPECTED_MARKER_ID in detected_ids
    )

    if expected_found:
        status_text = (
            f"PASS: ArUco ID {EXPECTED_MARKER_ID} detected"
        )
        status_colour = (0, 255, 0)
    else:
        status_text = (
            f"FAIL: ArUco ID {EXPECTED_MARKER_ID} not detected"
        )
        status_colour = (0, 0, 255)

    cv2.putText(
        annotated,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_colour,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated,
        f"Rejected candidates: {len(rejected)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / f"{image_path.stem}_detected.jpg"
    )

    saved = cv2.imwrite(
        str(output_path),
        annotated,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    if not saved:
        raise RuntimeError(
            f"Could not save result: {output_path}"
        )

    if expected_found:
        print(
            f"[PASS] {image_path.name} | "
            f"IDs: {detected_ids} | "
            f"Marker side: "
            f"{expected_marker_side_pixels:.2f} px"
        )
    else:
        print(
            f"[FAIL] {image_path.name} | "
            f"Detected IDs: {detected_ids or 'None'} | "
            f"Rejected candidates: {len(rejected)}"
        )

    return (
        expected_found,
        detected_ids,
        expected_marker_side_pixels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test ArUco DICT_4X4_50 marker ID 8 detection "
            "on one image or an entire folder."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to an image or folder containing "
            "undistorted images."
        ),
    )

    parser.add_argument(
        "--output",
        default="measurement/aruco_detection_results",
        help="Folder for annotated detection results.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_directory = Path(args.output)

    image_paths = get_input_images(input_path)

    dictionary, detector, parameters = create_detector()

    print("========================================")
    print("ARUCO DETECTION TEST")
    print("========================================")
    print("Dictionary: DICT_4X4_50")
    print(f"Expected marker ID: {EXPECTED_MARKER_ID}")
    print(f"Actual marker size: {MARKER_SIZE_MM} mm")
    print(f"Images found: {len(image_paths)}")
    print()

    passed = 0
    failed = 0
    detected_side_lengths: list[float] = []

    for image_path in image_paths:
        (
            expected_found,
            _,
            side_pixels,
        ) = process_image(
            image_path=image_path,
            output_directory=output_directory,
            dictionary=dictionary,
            detector=detector,
            parameters=parameters,
        )

        if expected_found:
            passed += 1

            if side_pixels is not None:
                detected_side_lengths.append(
                    side_pixels
                )
        else:
            failed += 1

    print()
    print("========================================")
    print("DETECTION SUMMARY")
    print("========================================")
    print(f"Total images: {len(image_paths)}")
    print(f"Marker detected: {passed}")
    print(f"Marker not detected: {failed}")

    detection_rate = (
        passed / len(image_paths) * 100
    )

    print(f"Detection rate: {detection_rate:.2f}%")

    if detected_side_lengths:
        mean_marker_pixels = float(
            np.mean(detected_side_lengths)
        )

        print(
            f"Mean detected marker side: "
            f"{mean_marker_pixels:.2f} pixels"
        )

    print()
    print(
        f"Annotated results: "
        f"{output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()