from __future__ import annotations

import re
import uuid
from pathlib import Path


MAIN_FOLDER = Path("dataset/raw/main_only")
ARUCO_FOLDER = Path("dataset/raw/with_aruco")

SUPPORTED_EXTENSIONS = {
    ".heic",
    ".heif",
}


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


def get_images(folder: Path) -> list[Path]:
    """
    Return all supported image files in natural filename order.
    """

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {folder.resolve()}"
        )

    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=natural_sort_key,
    )


def rename_images(
    folder: Path,
    prefix: str,
) -> int:
    """
    Rename all HEIC/HEIF images using a safe two-stage process.

    The temporary rename prevents filename collisions.
    """

    images = get_images(folder)

    if not images:
        print(f"[WARNING] No HEIC images found in: {folder}")
        return 0

    print()
    print(f"Folder: {folder.resolve()}")
    print(f"Images found: {len(images)}")
    print(f"New prefix: {prefix}")
    print()

    temporary_files: list[Path] = []
    temporary_id = uuid.uuid4().hex

    # First stage: move every file to a unique temporary name.
    for index, image_path in enumerate(images, start=1):
        temporary_path = folder / (
            f".rename_tmp_{temporary_id}_{index:04d}"
            f"{image_path.suffix.lower()}"
        )

        image_path.rename(temporary_path)
        temporary_files.append(temporary_path)

    # Second stage: assign the final sequential names.
    for index, temporary_path in enumerate(
        temporary_files,
        start=1,
    ):
        final_path = folder / f"{prefix}_{index:04d}.heic"

        if final_path.exists():
            raise FileExistsError(
                f"Target file already exists: {final_path}"
            )

        temporary_path.rename(final_path)

        print(
            f"[RENAMED] {temporary_path.name} "
            f"-> {final_path.name}"
        )

    return len(temporary_files)


def main() -> None:
    print("========================================")
    print("DATASET IMAGE RENAMER")
    print("========================================")

    main_count = rename_images(
        folder=MAIN_FOLDER,
        prefix="main",
    )

    aruco_count = rename_images(
        folder=ARUCO_FOLDER,
        prefix="aruco",
    )

    print()
    print("========================================")
    print("RENAMING COMPLETE")
    print("========================================")
    print(f"Main-only images renamed: {main_count}")
    print(f"ArUco images renamed: {aruco_count}")

    if main_count != 75:
        print(
            f"[WARNING] Expected 75 main-only images, "
            f"but found {main_count}."
        )

    if aruco_count != 45:
        print(
            f"[WARNING] Expected 45 ArUco images, "
            f"but found {aruco_count}."
        )


if __name__ == "__main__":
    main()