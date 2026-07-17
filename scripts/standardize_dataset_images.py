from __future__ import annotations

from pathlib import Path

import cv2


# ---------------------------------------------------------
# Input folders
# ---------------------------------------------------------

MAIN_INPUT_FOLDER = Path("dataset/jpeg/main_only")
ARUCO_INPUT_FOLDER = Path("dataset/jpeg/with_aruco")


# ---------------------------------------------------------
# Output folders
# ---------------------------------------------------------

MAIN_OUTPUT_FOLDER = Path("dataset/standardized/main_only")
ARUCO_OUTPUT_FOLDER = Path("dataset/standardized/with_aruco")


# Calibration resolution: width × height
TARGET_WIDTH = 960
TARGET_HEIGHT = 1280

JPEG_QUALITY = 95

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
}


def get_images(folder: Path) -> list[Path]:
    """Return JPEG images in filename order."""

    if not folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {folder.resolve()}"
        )

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def standardize_folder(
    input_folder: Path,
    output_folder: Path,
) -> tuple[int, int, int]:
    """
    Rotate landscape images into portrait orientation and resize
    all images to the camera-calibration resolution.
    """

    image_paths = get_images(input_folder)

    if not image_paths:
        print(f"[WARNING] No JPEG images found in {input_folder}")
        return 0, 0, 0

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful = 0
    failed = 0
    rotated = 0

    print()
    print("==================================================")
    print(f"Input:  {input_folder.resolve()}")
    print(f"Output: {output_folder.resolve()}")
    print(f"Images found: {len(image_paths)}")
    print("==================================================")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[FAILED TO READ] {image_path.name}")
            failed += 1
            continue

        original_height, original_width = image.shape[:2]

        # Rotate landscape images into portrait orientation.
        if original_width > original_height:
            image = cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE,
            )

            rotated += 1

        current_height, current_width = image.shape[:2]

        # Check that the image is now portrait.
        if current_width >= current_height:
            print(
                f"[INVALID ORIENTATION] {image_path.name}: "
                f"{current_width} × {current_height}"
            )
            failed += 1
            continue

        # Check for the expected 3:4 aspect ratio.
        current_ratio = current_width / current_height
        target_ratio = TARGET_WIDTH / TARGET_HEIGHT

        if abs(current_ratio - target_ratio) > 0.001:
            print(
                f"[WRONG ASPECT RATIO] {image_path.name}: "
                f"{current_width} × {current_height}"
            )
            failed += 1
            continue

        # INTER_AREA is suitable for downscaling.
        standardized = cv2.resize(
            image,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        output_path = output_folder / image_path.name

        saved = cv2.imwrite(
            str(output_path),
            standardized,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY,
            ],
        )

        if not saved:
            print(f"[FAILED TO SAVE] {image_path.name}")
            failed += 1
            continue

        successful += 1

        action = (
            "ROTATED + RESIZED"
            if original_width > original_height
            else "RESIZED"
        )

        print(
            f"[{action}] {image_path.name}: "
            f"{original_width} × {original_height} "
            f"-> {TARGET_WIDTH} × {TARGET_HEIGHT}"
        )

    return successful, failed, rotated


def main() -> None:
    print("========================================")
    print("DATASET IMAGE STANDARDIZATION")
    print("========================================")
    print(
        f"Target resolution: "
        f"{TARGET_WIDTH} × {TARGET_HEIGHT}"
    )

    (
        main_successful,
        main_failed,
        main_rotated,
    ) = standardize_folder(
        MAIN_INPUT_FOLDER,
        MAIN_OUTPUT_FOLDER,
    )

    (
        aruco_successful,
        aruco_failed,
        aruco_rotated,
    ) = standardize_folder(
        ARUCO_INPUT_FOLDER,
        ARUCO_OUTPUT_FOLDER,
    )

    print()
    print("========================================")
    print("STANDARDIZATION COMPLETE")
    print("========================================")
    print(f"Main images processed: {main_successful}")
    print(f"Main images rotated:   {main_rotated}")
    print(f"Main failures:         {main_failed}")
    print()
    print(f"ArUco images processed: {aruco_successful}")
    print(f"ArUco images rotated:   {aruco_rotated}")
    print(f"ArUco failures:         {aruco_failed}")


if __name__ == "__main__":
    main()