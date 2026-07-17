from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from measure_object import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_MARKER_ID,
    DEFAULT_MARKER_SIZE_MM,
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    SUPPORTED_IMAGE_EXTENSIONS,
    build_image_to_metric_homography,
    detect_marker,
    find_object_contour,
    keep_largest_component,
    load_calibration,
    load_model,
    measure_metric_contour,
    predict_mask,
    prepare_image,
    transform_contour_to_metric_plane,
    undistort_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "dataset"
    / "undistorted"
    / "measurement_validation"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "measurement"
    / "outputs"
)

DEFAULT_ACTUAL_LONG_MM = 110.0
DEFAULT_ACTUAL_SHORT_MM = 110.0


def list_images(
    input_directory: Path,
) -> list[Path]:
    if not input_directory.exists():
        raise FileNotFoundError(
            f"Input directory not found:\n{input_directory}"
        )

    image_paths = sorted(
        path
        for path in input_directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in:\n{input_directory}"
        )

    return image_paths


def calculate_error_metrics(
    actual_value: float,
    predicted_value: float,
) -> tuple[float, float]:
    absolute_error = abs(
        predicted_value - actual_value
    )

    percentage_error = (
        absolute_error / actual_value * 100.0
        if actual_value != 0
        else 0.0
    )

    return absolute_error, percentage_error


def process_single_image(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    model_height: int,
    model_width: int,
    camera_matrix: np.ndarray | None,
    distortion_coefficients: np.ndarray | None,
    marker_id: int,
    marker_size_mm: float,
    threshold: float,
    already_undistorted: bool,
    actual_long_mm: float,
    actual_short_mm: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image": image_path.name,
        "status": "failed",
        "actual_long_mm": actual_long_mm,
        "predicted_long_mm": "",
        "absolute_error_long_mm": "",
        "percentage_error_long": "",
        "actual_short_mm": actual_short_mm,
        "predicted_short_mm": "",
        "absolute_error_short_mm": "",
        "percentage_error_short": "",
        "overall_absolute_error_mm": "",
        "overall_percentage_error": "",
        "confidence": "",
        "failure_reason": "",
    }

    try:
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                "OpenCV could not read the image."
            )

        if already_undistorted:
            working_image = image
        else:
            if (
                camera_matrix is None
                or distortion_coefficients is None
            ):
                raise RuntimeError(
                    "Calibration data was not loaded."
                )

            working_image = undistort_image(
                image,
                camera_matrix,
                distortion_coefficients,
            )

        marker_corners = detect_marker(
            working_image,
            marker_id,
        )

        image_to_metric_homography = (
            build_image_to_metric_homography(
                marker_corners,
                marker_size_mm,
            )
        )

        output_height, output_width = (
            working_image.shape[:2]
        )

        image_tensor = prepare_image(
            working_image,
            model_height,
            model_width,
        ).to(device)

        probability_map, binary_mask = predict_mask(
            model,
            image_tensor,
            output_height,
            output_width,
            threshold,
        )

        binary_mask = keep_largest_component(
            binary_mask
        )

        contour = find_object_contour(
            binary_mask
        )

        metric_contour = (
            transform_contour_to_metric_plane(
                contour,
                image_to_metric_homography,
            )
        )

        measurement = measure_metric_contour(
            metric_contour
        )

        predicted_long_mm = float(
            measurement["long_side_mm"]
        )

        predicted_short_mm = float(
            measurement["short_side_mm"]
        )

        long_absolute_error, long_percentage_error = (
            calculate_error_metrics(
                actual_long_mm,
                predicted_long_mm,
            )
        )

        short_absolute_error, short_percentage_error = (
            calculate_error_metrics(
                actual_short_mm,
                predicted_short_mm,
            )
        )

        overall_absolute_error = float(
            np.mean(
                [
                    long_absolute_error,
                    short_absolute_error,
                ]
            )
        )

        overall_percentage_error = float(
            np.mean(
                [
                    long_percentage_error,
                    short_percentage_error,
                ]
            )
        )

        mask_pixels = binary_mask > 0

        confidence = (
            float(
                probability_map[
                    mask_pixels
                ].mean()
            )
            if mask_pixels.any()
            else 0.0
        )

        row.update(
            {
                "status": "success",
                "predicted_long_mm": predicted_long_mm,
                "absolute_error_long_mm": (
                    long_absolute_error
                ),
                "percentage_error_long": (
                    long_percentage_error
                ),
                "predicted_short_mm": predicted_short_mm,
                "absolute_error_short_mm": (
                    short_absolute_error
                ),
                "percentage_error_short": (
                    short_percentage_error
                ),
                "overall_absolute_error_mm": (
                    overall_absolute_error
                ),
                "overall_percentage_error": (
                    overall_percentage_error
                ),
                "confidence": confidence,
                "failure_reason": "",
            }
        )

    except Exception as error:
        row["failure_reason"] = str(error)

    return row


def calculate_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    successful_rows = [
        row
        for row in rows
        if row["status"] == "success"
    ]

    failed_rows = [
        row
        for row in rows
        if row["status"] == "failed"
    ]

    summary: dict[str, Any] = {
        "total_images": len(rows),
        "successful_images": len(
            successful_rows
        ),
        "failed_images": len(
            failed_rows
        ),
        "success_rate_percent": (
            len(successful_rows)
            / len(rows)
            * 100.0
            if rows
            else 0.0
        ),
        "long_side_mae_mm": None,
        "short_side_mae_mm": None,
        "overall_mae_mm": None,
        "long_side_mpe_percent": None,
        "short_side_mpe_percent": None,
        "overall_mpe_percent": None,
        "mean_confidence_percent": None,
    }

    if not successful_rows:
        return summary

    summary.update(
        {
            "long_side_mae_mm": float(
                np.mean(
                    [
                        row[
                            "absolute_error_long_mm"
                        ]
                        for row in successful_rows
                    ]
                )
            ),
            "short_side_mae_mm": float(
                np.mean(
                    [
                        row[
                            "absolute_error_short_mm"
                        ]
                        for row in successful_rows
                    ]
                )
            ),
            "overall_mae_mm": float(
                np.mean(
                    [
                        row[
                            "overall_absolute_error_mm"
                        ]
                        for row in successful_rows
                    ]
                )
            ),
            "long_side_mpe_percent": float(
                np.mean(
                    [
                        row[
                            "percentage_error_long"
                        ]
                        for row in successful_rows
                    ]
                )
            ),
            "short_side_mpe_percent": float(
                np.mean(
                    [
                        row[
                            "percentage_error_short"
                        ]
                        for row in successful_rows
                    ]
                )
            ),
            "overall_mpe_percent": float(
                np.mean(
                    [
                        row[
                            "overall_percentage_error"
                        ]
                        for row in successful_rows
                    ]
                )
            ),
            "mean_confidence_percent": float(
                np.mean(
                    [
                        row["confidence"] * 100.0
                        for row in successful_rows
                    ]
                )
            ),
        }
    )

    return summary


def save_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "image",
        "status",
        "actual_long_mm",
        "predicted_long_mm",
        "absolute_error_long_mm",
        "percentage_error_long",
        "actual_short_mm",
        "predicted_short_mm",
        "absolute_error_short_mm",
        "percentage_error_short",
        "overall_absolute_error_mm",
        "overall_percentage_error",
        "confidence",
        "failure_reason",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(
    output_path: Path,
    summary: dict[str, Any],
) -> None:
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )


def print_summary(
    summary: dict[str, Any],
) -> None:
    print()
    print("========================================")
    print("MEASUREMENT VALIDATION COMPLETE")
    print("========================================")
    print(
        f"Total images:          "
        f"{summary['total_images']}"
    )
    print(
        f"Successful images:     "
        f"{summary['successful_images']}"
    )
    print(
        f"Failed images:         "
        f"{summary['failed_images']}"
    )
    print(
        f"Success rate:          "
        f"{summary['success_rate_percent']:.2f}%"
    )

    if summary["successful_images"] == 0:
        print(
            "No successful measurements were available "
            "for metric calculation."
        )
        return

    print()
    print(
        f"Long-side MAE:         "
        f"{summary['long_side_mae_mm']:.4f} mm"
    )
    print(
        f"Short-side MAE:        "
        f"{summary['short_side_mae_mm']:.4f} mm"
    )
    print(
        f"Overall MAE:           "
        f"{summary['overall_mae_mm']:.4f} mm"
    )
    print()
    print(
        f"Long-side MPE:         "
        f"{summary['long_side_mpe_percent']:.4f}%"
    )
    print(
        f"Short-side MPE:        "
        f"{summary['short_side_mpe_percent']:.4f}%"
    )
    print(
        f"Overall MPE:           "
        f"{summary['overall_mpe_percent']:.4f}%"
    )
    print()
    print(
        f"Mean confidence:       "
        f"{summary['mean_confidence_percent']:.2f}%"
    )


def run_validation(
    input_directory: Path,
    output_directory: Path,
    model_path: Path,
    calibration_path: Path,
    marker_id: int,
    marker_size_mm: float,
    threshold: float,
    actual_long_mm: float,
    actual_short_mm: float,
    already_undistorted: bool,
) -> None:
    image_paths = list_images(
        input_directory
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, model_height, model_width = (
        load_model(
            model_path,
            device,
        )
    )

    camera_matrix = None
    distortion_coefficients = None

    if not already_undistorted:
        (
            camera_matrix,
            distortion_coefficients,
        ) = load_calibration(
            calibration_path
        )

    rows: list[dict[str, Any]] = []

    print("========================================")
    print("STARTING MEASUREMENT VALIDATION")
    print("========================================")
    print(f"Device:              {device}")
    print(f"Images found:        {len(image_paths)}")
    print(f"Input directory:     {input_directory}")
    print(
        f"Actual dimensions:   "
        f"{actual_long_mm:.2f} x "
        f"{actual_short_mm:.2f} mm"
    )
    print()

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(image_paths):02d}] "
            f"{image_path.name}"
        )

        row = process_single_image(
            image_path=image_path,
            model=model,
            device=device,
            model_height=model_height,
            model_width=model_width,
            camera_matrix=camera_matrix,
            distortion_coefficients=(
                distortion_coefficients
            ),
            marker_id=marker_id,
            marker_size_mm=marker_size_mm,
            threshold=threshold,
            already_undistorted=(
                already_undistorted
            ),
            actual_long_mm=actual_long_mm,
            actual_short_mm=actual_short_mm,
        )

        rows.append(row)

        if row["status"] == "success":
            print(
                "  Success: "
                f"{row['predicted_long_mm']:.2f} x "
                f"{row['predicted_short_mm']:.2f} mm | "
                f"error {row['overall_absolute_error_mm']:.2f} mm"
            )
        else:
            print(
                "  Failed: "
                f"{row['failure_reason']}"
            )

    summary = calculate_summary(rows)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_directory
        / "measurement_validation.csv"
    )

    summary_path = (
        output_directory
        / "measurement_validation_summary.json"
    )

    save_csv(
        csv_path,
        rows,
    )

    save_summary_json(
        summary_path,
        summary,
    )

    print_summary(summary)

    print()
    print("Saved outputs:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate object measurements across a directory "
            "of ArUco reference images."
        )
    )

    parser.add_argument(
        "--input-directory",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
        help=(
            "Directory containing measurement-validation images."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for CSV and JSON outputs.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to best_unet.pt.",
    )

    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="Path to camera_calibration.npz.",
    )

    parser.add_argument(
        "--marker-id",
        type=int,
        default=DEFAULT_MARKER_ID,
        help="Expected ArUco marker ID.",
    )

    parser.add_argument(
        "--marker-size-mm",
        type=float,
        default=DEFAULT_MARKER_SIZE_MM,
        help="Physical ArUco side length in millimetres.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Segmentation probability threshold.",
    )

    parser.add_argument(
        "--actual-long-mm",
        type=float,
        default=DEFAULT_ACTUAL_LONG_MM,
        help="Ground-truth long-side length.",
    )

    parser.add_argument(
        "--actual-short-mm",
        type=float,
        default=DEFAULT_ACTUAL_SHORT_MM,
        help="Ground-truth short-side length.",
    )

    parser.add_argument(
        "--already-undistorted",
        action="store_true",
        help=(
            "Skip camera undistortion because the input "
            "directory already contains undistorted images."
        ),
    )

    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    if args.marker_size_mm <= 0:
        raise ValueError(
            "--marker-size-mm must be greater than zero."
        )

    if (
        args.actual_long_mm <= 0
        or args.actual_short_mm <= 0
    ):
        raise ValueError(
            "Actual dimensions must be greater than zero."
        )

    run_validation(
        input_directory=args.input_directory,
        output_directory=args.output_directory,
        model_path=args.model,
        calibration_path=args.calibration,
        marker_id=args.marker_id,
        marker_size_mm=args.marker_size_mm,
        threshold=args.threshold,
        actual_long_mm=args.actual_long_mm,
        actual_short_mm=args.actual_short_mm,
        already_undistorted=args.already_undistorted,
    )


if __name__ == "__main__":
    main()