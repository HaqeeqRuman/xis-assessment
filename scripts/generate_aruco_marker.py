from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_50

# Keep this at 40 mm for reliable detection and measurement.
MARKER_SIZE_MM = 40.0

# Resolution of generated marker image.
MARKER_PIXELS = 1000

# White space around the marker.
WHITE_MARGIN_MM = 25.0

OUTPUT_DIRECTORY = Path("measurement/reference_marker")


def generate_marker(
    dictionary,
    marker_id: int,
) -> np.ndarray:
    """Generate one marker image."""

    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(
            dictionary,
            marker_id,
            MARKER_PIXELS,
            borderBits=1,
        )

    if hasattr(cv2.aruco, "drawMarker"):
        marker = np.zeros(
            (MARKER_PIXELS, MARKER_PIXELS),
            dtype=np.uint8,
        )

        cv2.aruco.drawMarker(
            dictionary,
            marker_id,
            MARKER_PIXELS,
            marker,
            1,
        )

        return marker

    raise RuntimeError(
        "This OpenCV installation does not support "
        "ArUco marker generation."
    )


def calculate_black_percentage(marker: np.ndarray) -> float:
    """
    Calculate how much of the marker image is black.

    A lower percentage means less printer ink is required.
    """

    black_pixels = np.count_nonzero(marker < 128)
    total_pixels = marker.size

    return float(
        black_pixels / total_pixels * 100
    )


def find_lowest_ink_marker(
    dictionary,
) -> tuple[int, np.ndarray, float]:
    """
    Search every marker in the dictionary and return the marker
    requiring the least black ink.
    """

    marker_count = dictionary.bytesList.shape[0]

    best_marker_id: int | None = None
    best_marker: np.ndarray | None = None
    best_black_percentage = float("inf")

    print("Checking marker ink usage...")

    for marker_id in range(marker_count):
        marker = generate_marker(
            dictionary,
            marker_id,
        )

        black_percentage = calculate_black_percentage(
            marker
        )

        print(
            f"Marker ID {marker_id:02d}: "
            f"{black_percentage:.2f}% black"
        )

        if black_percentage < best_black_percentage:
            best_marker_id = marker_id
            best_marker = marker
            best_black_percentage = black_percentage

    if best_marker_id is None or best_marker is None:
        raise RuntimeError(
            "Could not select an ArUco marker."
        )

    return (
        best_marker_id,
        best_marker,
        best_black_percentage,
    )


def save_marker_png(
    marker: np.ndarray,
    output_path: Path,
) -> None:
    """Save the generated marker as PNG."""

    saved = cv2.imwrite(
        str(output_path),
        marker,
    )

    if not saved:
        raise RuntimeError(
            f"Could not save PNG: {output_path}"
        )


def create_printable_pdf(
    marker_id: int,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Create an A4 PDF containing only the marker."""

    page_width, page_height = A4

    marker_size_points = MARKER_SIZE_MM * mm
    white_margin_points = WHITE_MARGIN_MM * mm

    backing_size = (
        marker_size_points
        + 2 * white_margin_points
    )

    backing_x = (
        page_width - backing_size
    ) / 2

    backing_y = (
        page_height - backing_size
    ) / 2

    marker_x = backing_x + white_margin_points
    marker_y = backing_y + white_margin_points

    pdf = canvas.Canvas(
        str(pdf_path),
        pagesize=A4,
    )

    pdf.setTitle(
        f"ArUco Marker ID {marker_id}"
    )

    # White area around the marker.
    pdf.setFillColorRGB(1, 1, 1)

    pdf.rect(
        backing_x,
        backing_y,
        backing_size,
        backing_size,
        fill=1,
        stroke=0,
    )

    # Draw only the ArUco marker.
    pdf.drawImage(
        ImageReader(str(png_path)),
        marker_x,
        marker_y,
        width=marker_size_points,
        height=marker_size_points,
        preserveAspectRatio=True,
        mask="auto",
    )

    pdf.save()


def main() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "The cv2.aruco module is unavailable. Install it with:\n"
            "python -m pip uninstall opencv-python -y\n"
            "python -m pip install opencv-contrib-python"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dictionary = cv2.aruco.getPredefinedDictionary(
        ARUCO_DICTIONARY_ID
    )

    (
        marker_id,
        marker,
        black_percentage,
    ) = find_lowest_ink_marker(dictionary)

    png_output = (
        OUTPUT_DIRECTORY
        / f"aruco_low_ink_id_{marker_id}.png"
    )

    pdf_output = (
        OUTPUT_DIRECTORY
        / (
            f"aruco_low_ink_id_{marker_id}_"
            f"{MARKER_SIZE_MM:g}mm.pdf"
        )
    )

    save_marker_png(
        marker,
        png_output,
    )

    create_printable_pdf(
        marker_id,
        png_output,
        pdf_output,
    )

    print()
    print("========================================")
    print("LOW-INK ARUCO MARKER GENERATED")
    print("========================================")
    print("Dictionary: DICT_4X4_50")
    print(f"Selected marker ID: {marker_id}")
    print(
        f"Black area: {black_percentage:.2f}%"
    )
    print(
        f"Printed size: "
        f"{MARKER_SIZE_MM:g} mm × "
        f"{MARKER_SIZE_MM:g} mm"
    )
    print()
    print(f"PNG: {png_output.resolve()}")
    print(f"PDF: {pdf_output.resolve()}")
    print()
    print(
        "Remember the selected marker ID because the "
        "detection code must look for this same ID."
    )


if __name__ == "__main__":
    main()