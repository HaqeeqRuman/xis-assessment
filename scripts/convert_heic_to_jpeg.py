from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


# Allow Pillow to open HEIC and HEIF files.
register_heif_opener(thumbnails=False)


# ---------------------------------------------------------
# Input folders containing the original HEIC images
# ---------------------------------------------------------

MAIN_INPUT_FOLDER = Path("dataset/raw/main_only")
ARUCO_INPUT_FOLDER = Path("dataset/raw/with_aruco")


# ---------------------------------------------------------
# Output folders for converted JPEG images
# ---------------------------------------------------------

MAIN_OUTPUT_FOLDER = Path("dataset/jpeg/main_only")
ARUCO_OUTPUT_FOLDER = Path("dataset/jpeg/with_aruco")


SUPPORTED_EXTENSIONS = {
    ".heic",
    ".heif",
    ".heics",
    ".heifs",
    ".hif",
}

JPEG_QUALITY = 95

# Set to True only when you intentionally want existing
# JPEG files to be replaced.
OVERWRITE_EXISTING = False


def natural_sort_key(path: Path) -> list[object]:
    """
    Sort filenames naturally.

    Example:
    IMG_2.HEIC comes before IMG_10.HEIC.
    """

    parts = re.split(r"(\d+)", path.name.lower())

    return [
        int(part) if part.isdigit() else part
        for part in parts
    ]


def get_heic_images(folder: Path) -> list[Path]:
    """
    Return all supported HEIC/HEIF images in natural order.
    """

    if not folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {folder.resolve()}"
        )

    images = sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=natural_sort_key,
    )

    return images


def convert_folder(
    input_folder: Path,
    output_folder: Path,
    prefix: str,
) -> tuple[int, int]:
    """
    Convert every HEIC/HEIF image in a folder to JPEG.

    Files are renamed sequentially, for example:
    main_0001.jpg
    main_0002.jpg
    """

    image_paths = get_heic_images(input_folder)

    if not image_paths:
        print(
            f"[WARNING] No HEIC/HEIF images found in: "
            f"{input_folder.resolve()}"
        )
        return 0, 0

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    converted_count = 0
    failed_count = 0

    print()
    print("--------------------------------------------------")
    print(f"Input:  {input_folder.resolve()}")
    print(f"Output: {output_folder.resolve()}")
    print(f"Images found: {len(image_paths)}")
    print("--------------------------------------------------")

    for index, input_path in enumerate(
        image_paths,
        start=1,
    ):
        output_path = (
            output_folder
            / f"{prefix}_{index:04d}.jpg"
        )

        if output_path.exists() and not OVERWRITE_EXISTING:
            print(
                f"[SKIPPED] {output_path.name} already exists"
            )
            continue

        try:
            with Image.open(input_path) as image:
                # Apply the phone camera's EXIF orientation so the
                # saved JPEG has the correct physical orientation.
                image = ImageOps.exif_transpose(image)

                # JPEG does not support alpha transparency.
                if image.mode != "RGB":
                    image = image.convert("RGB")

                original_size = image.size

                image.save(
                    output_path,
                    format="JPEG",
                    quality=JPEG_QUALITY,
                    subsampling=0,
                    optimize=True,
                )

            # Verify that the converted image can be opened.
            with Image.open(output_path) as converted_image:
                converted_size = converted_image.size

            if converted_size != original_size:
                raise RuntimeError(
                    "Image dimensions changed during conversion: "
                    f"{original_size} -> {converted_size}"
                )

            converted_count += 1

            print(
                f"[CONVERTED] {input_path.name} "
                f"-> {output_path.name} "
                f"({converted_size[0]} x {converted_size[1]})"
            )

        except Exception as error:
            failed_count += 1

            print(
                f"[FAILED] {input_path.name}: {error}"
            )

            if output_path.exists():
                output_path.unlink()

    return converted_count, failed_count


def count_jpegs(folder: Path) -> int:
    """
    Count JPEG files in a folder.
    """

    if not folder.exists():
        return 0

    return len(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg"}
        ]
    )


def main() -> None:
    print("========================================")
    print("HEIC TO JPEG DATASET CONVERTER")
    print("========================================")
    print(
        "Original HEIC files will remain unchanged."
    )
    print(
        f"JPEG quality: {JPEG_QUALITY}"
    )

    main_converted, main_failed = convert_folder(
        input_folder=MAIN_INPUT_FOLDER,
        output_folder=MAIN_OUTPUT_FOLDER,
        prefix="main",
    )

    aruco_converted, aruco_failed = convert_folder(
        input_folder=ARUCO_INPUT_FOLDER,
        output_folder=ARUCO_OUTPUT_FOLDER,
        prefix="aruco",
    )

    main_total = count_jpegs(
        MAIN_OUTPUT_FOLDER
    )

    aruco_total = count_jpegs(
        ARUCO_OUTPUT_FOLDER
    )

    print()
    print("========================================")
    print("CONVERSION COMPLETE")
    print("========================================")
    print(
        f"Main images converted this run: "
        f"{main_converted}"
    )
    print(
        f"ArUco images converted this run: "
        f"{aruco_converted}"
    )
    print(
        f"Main conversion failures: "
        f"{main_failed}"
    )
    print(
        f"ArUco conversion failures: "
        f"{aruco_failed}"
    )
    print()
    print(
        f"Total main JPEG images: {main_total}"
    )
    print(
        f"Total ArUco JPEG images: {aruco_total}"
    )

    if main_total != 75:
        print(
            f"[WARNING] Expected 75 main images, "
            f"but the JPEG folder contains {main_total}."
        )

    if aruco_total != 45:
        print(
            f"[WARNING] Expected 45 ArUco images, "
            f"but the JPEG folder contains {aruco_total}."
        )

    if main_failed == 0 and aruco_failed == 0:
        print()
        print(
            "All discovered images were converted successfully."
        )

    print()
    print("Main JPEG folder:")
    print(MAIN_OUTPUT_FOLDER.resolve())
    print()
    print("ArUco JPEG folder:")
    print(ARUCO_OUTPUT_FOLDER.resolve())


if __name__ == "__main__":
    main()