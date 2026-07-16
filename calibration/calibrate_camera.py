from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


# Your printed board has 10 × 8 squares,
# therefore it has 9 × 7 INNER corners.
CHECKERBOARD = (9, 7)

# Physical printed square size.
SQUARE_SIZE_MM = 20.0

IMAGE_DIRECTORY = Path("calibration/images")
PREVIEW_DIRECTORY = Path("calibration/detected_corners")
OUTPUT_DIRECTORY = Path("calibration/outputs")


def get_image_paths(directory: Path) -> list[Path]:
    """Return all supported calibration images in filename order."""

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )


def calculate_reprojection_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rotation_vectors,
    translation_vectors,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> list[float]:
    """
    Calculate reprojection RMSE separately for every calibration image.

    OpenCV may return detected and projected points with different shapes:

        detected:  (N, 1, 2) or (N, 2)
        projected: (N, 1, 2)

    Both arrays are converted to (N, 2) float64 arrays before the
    error is calculated.
    """

    errors: list[float] = []

    for index, world_points in enumerate(object_points):
        projected_points, _ = cv2.projectPoints(
            world_points,
            rotation_vectors[index],
            translation_vectors[index],
            camera_matrix,
            distortion_coefficients,
        )

        detected = np.asarray(
            image_points[index],
            dtype=np.float64,
        ).reshape(-1, 2)

        projected = np.asarray(
            projected_points,
            dtype=np.float64,
        ).reshape(-1, 2)

        if detected.shape != projected.shape:
            raise ValueError(
                f"Corner shape mismatch for calibration image "
                f"{index + 1}: detected={detected.shape}, "
                f"projected={projected.shape}"
            )

        # Euclidean distance between each detected corner and its
        # corresponding projected corner.
        corner_distances = np.linalg.norm(
            detected - projected,
            axis=1,
        )

        # Root Mean Square Error for this calibration image.
        image_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(corner_distances)
                )
            )
        )

        errors.append(image_rmse)

    return errors


def main() -> None:
    if not IMAGE_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Image directory not found: "
            f"{IMAGE_DIRECTORY.resolve()}"
        )

    PREVIEW_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = get_image_paths(IMAGE_DIRECTORY)

    if len(image_paths) < 20:
        raise ValueError(
            "At least 20 calibration images are needed. "
            f"Only {len(image_paths)} were found."
        )

    print("========================================")
    print("CAMERA CALIBRATION")
    print("========================================")
    print(f"Images found: {len(image_paths)}")
    print(
        f"Checkerboard inner corners: "
        f"{CHECKERBOARD[0]} × {CHECKERBOARD[1]}"
    )
    print(f"Square size: {SQUARE_SIZE_MM} mm")
    print()

    # Real-world checkerboard corner coordinates.
    #
    # For a 9 × 7 inner-corner board with 20 mm squares:
    #
    # (0, 0, 0)
    # (20, 0, 0)
    # (40, 0, 0)
    # ...
    object_point_template = np.zeros(
        (
            CHECKERBOARD[0] * CHECKERBOARD[1],
            3,
        ),
        dtype=np.float32,
    )

    object_point_template[:, :2] = (
        np.mgrid[
            0:CHECKERBOARD[0],
            0:CHECKERBOARD[1],
        ]
        .T.reshape(-1, 2)
        .astype(np.float32)
    )

    object_point_template *= SQUARE_SIZE_MM

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    successful_images: list[str] = []
    failed_images: list[str] = []

    image_size: tuple[int, int] | None = None

    corner_criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.001,
    )

    detection_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[FAILED TO READ] {image_path.name}")
            failed_images.append(image_path.name)
            continue

        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        current_size = (
            grayscale.shape[1],
            grayscale.shape[0],
        )

        if image_size is None:
            image_size = current_size

        elif current_size != image_size:
            print(
                f"[WRONG RESOLUTION] {image_path.name}: "
                f"{current_size[0]} × {current_size[1]}, "
                f"expected "
                f"{image_size[0]} × {image_size[1]}"
            )

            failed_images.append(image_path.name)
            continue

        found, corners = cv2.findChessboardCorners(
            grayscale,
            CHECKERBOARD,
            detection_flags,
        )

        if not found:
            print(f"[NOT DETECTED] {image_path.name}")
            failed_images.append(image_path.name)
            continue

        refined_corners = cv2.cornerSubPix(
            grayscale,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=corner_criteria,
        )

        object_points.append(
            object_point_template.copy()
        )

        image_points.append(refined_corners)
        successful_images.append(image_path.name)

        preview = image.copy()

        cv2.drawChessboardCorners(
            preview,
            CHECKERBOARD,
            refined_corners,
            found,
        )

        preview_path = (
            PREVIEW_DIRECTORY / image_path.name
        )

        saved = cv2.imwrite(
            str(preview_path),
            preview,
        )

        if not saved:
            print(
                f"[WARNING] Could not save preview: "
                f"{preview_path}"
            )

        print(f"[DETECTED] {image_path.name}")

    if image_size is None:
        raise RuntimeError(
            "No readable calibration images were found."
        )

    if len(successful_images) < 20:
        raise RuntimeError(
            f"Only {len(successful_images)} checkerboards "
            "were detected successfully. At least 20 "
            "successful calibration images are required."
        )

    print()
    print("Running intrinsic camera calibration...")

    (
        calibration_rms,
        camera_matrix,
        distortion_coefficients,
        rotation_vectors,
        translation_vectors,
    ) = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    per_image_errors = calculate_reprojection_errors(
        object_points=object_points,
        image_points=image_points,
        rotation_vectors=rotation_vectors,
        translation_vectors=translation_vectors,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
    )

    mean_error = float(
        np.mean(per_image_errors)
    )

    median_error = float(
        np.median(per_image_errors)
    )

    maximum_error = float(
        np.max(per_image_errors)
    )

    minimum_error = float(
        np.min(per_image_errors)
    )

    maximum_error_index = int(
        np.argmax(per_image_errors)
    )

    maximum_error_image = successful_images[
        maximum_error_index
    ]

    # Save parameters in NumPy format for later undistortion
    # and measurement scripts.
    calibration_file = (
        OUTPUT_DIRECTORY / "camera_calibration.npz"
    )

    np.savez(
        calibration_file,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        image_width=image_size[0],
        image_height=image_size[1],
        square_size_mm=SQUARE_SIZE_MM,
        checkerboard_columns=CHECKERBOARD[0],
        checkerboard_rows=CHECKERBOARD[1],
        calibration_rms=float(calibration_rms),
        mean_reprojection_error_px=mean_error,
        median_reprojection_error_px=median_error,
        maximum_reprojection_error_px=maximum_error,
        minimum_reprojection_error_px=minimum_error,
    )

    per_image_error_mapping = {
        name: float(error)
        for name, error in zip(
            successful_images,
            per_image_errors,
        )
    }

    # Highest-error images first, making inspection easier.
    sorted_per_image_errors = dict(
        sorted(
            per_image_error_mapping.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    report = {
        "checkerboard": {
            "square_columns": CHECKERBOARD[0] + 1,
            "square_rows": CHECKERBOARD[1] + 1,
            "inner_corner_columns": CHECKERBOARD[0],
            "inner_corner_rows": CHECKERBOARD[1],
            "square_size_mm": SQUARE_SIZE_MM,
        },
        "image_resolution": {
            "width": image_size[0],
            "height": image_size[1],
        },
        "image_counts": {
            "total": len(image_paths),
            "successful": len(successful_images),
            "failed": len(failed_images),
        },
        "successful_image_names": successful_images,
        "failed_image_names": failed_images,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": (
            distortion_coefficients
            .flatten()
            .tolist()
        ),
        "errors": {
            "opencv_calibration_rms_px": float(
                calibration_rms
            ),
            "mean_reprojection_error_px": mean_error,
            "median_reprojection_error_px": median_error,
            "minimum_reprojection_error_px": minimum_error,
            "maximum_reprojection_error_px": maximum_error,
            "maximum_error_image": maximum_error_image,
            "per_image_reprojection_rmse_px": (
                sorted_per_image_errors
            ),
        },
    }

    report_file = (
        OUTPUT_DIRECTORY / "calibration_report.json"
    )

    with report_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print()
    print("========================================")
    print("CAMERA CALIBRATION COMPLETE")
    print("========================================")
    print(f"Total images: {len(image_paths)}")
    print(
        f"Successful detections: "
        f"{len(successful_images)}"
    )
    print(
        f"Failed detections: "
        f"{len(failed_images)}"
    )
    print(
        f"Resolution: "
        f"{image_size[0]} × {image_size[1]}"
    )
    print()
    print(
        f"OpenCV calibration RMS: "
        f"{calibration_rms:.4f} px"
    )
    print(
        f"Mean reprojection RMSE: "
        f"{mean_error:.4f} px"
    )
    print(
        f"Median reprojection RMSE: "
        f"{median_error:.4f} px"
    )
    print(
        f"Minimum reprojection RMSE: "
        f"{minimum_error:.4f} px"
    )
    print(
        f"Maximum reprojection RMSE: "
        f"{maximum_error:.4f} px"
    )
    print(
        f"Highest-error image: "
        f"{maximum_error_image}"
    )

    print("\nCamera matrix:")
    print(camera_matrix)

    print("\nDistortion coefficients:")
    print(distortion_coefficients)

    print("\nSaved files:")
    print(calibration_file.resolve())
    print(report_file.resolve())

    print("\nDetected-corner previews:")
    print(PREVIEW_DIRECTORY.resolve())

    if mean_error < 0.3:
        print(
            "\nCalibration quality: EXCELLENT "
            "(mean error below 0.3 px)"
        )
    elif mean_error < 0.5:
        print(
            "\nCalibration quality: ACCEPTABLE/GOOD "
            "(mean error below 0.5 px)"
        )
    elif mean_error < 0.8:
        print(
            "\nCalibration quality: USABLE, but inspect "
            "high-error images and consider recalibrating."
        )
    else:
        print(
            "\nCalibration quality: POOR. Inspect the "
            "highest-error images and recalibrate."
        )


if __name__ == "__main__":
    main()