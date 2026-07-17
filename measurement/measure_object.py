from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "outputs"
    / "best_unet.pt"
)

DEFAULT_CALIBRATION_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "outputs"
    / "camera_calibration.npz"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "measurement"
    / "outputs"
)

DEFAULT_MARKER_ID = 8
DEFAULT_MARKER_SIZE_MM = 39.0
DEFAULT_THRESHOLD = 0.5

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


class DoubleConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(inputs)


class UNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        self.encoder1 = DoubleConv(
            input_channels,
            base_channels,
        )
        self.encoder2 = DoubleConv(
            base_channels,
            base_channels * 2,
        )
        self.encoder3 = DoubleConv(
            base_channels * 2,
            base_channels * 4,
        )
        self.encoder4 = DoubleConv(
            base_channels * 4,
            base_channels * 8,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.bottleneck = DoubleConv(
            base_channels * 8,
            base_channels * 16,
        )

        self.up4 = nn.ConvTranspose2d(
            base_channels * 16,
            base_channels * 8,
            kernel_size=2,
            stride=2,
        )
        self.decoder4 = DoubleConv(
            base_channels * 16,
            base_channels * 8,
        )

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2,
        )
        self.decoder3 = DoubleConv(
            base_channels * 8,
            base_channels * 4,
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )
        self.decoder2 = DoubleConv(
            base_channels * 4,
            base_channels * 2,
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )
        self.decoder1 = DoubleConv(
            base_channels * 2,
            base_channels,
        )

        self.output_layer = nn.Conv2d(
            base_channels,
            output_channels,
            kernel_size=1,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(
            self.pool(encoder1)
        )
        encoder3 = self.encoder3(
            self.pool(encoder2)
        )
        encoder4 = self.encoder4(
            self.pool(encoder3)
        )

        bottleneck = self.bottleneck(
            self.pool(encoder4)
        )

        decoder4 = self.up4(bottleneck)
        decoder4 = torch.cat(
            [decoder4, encoder4],
            dim=1,
        )
        decoder4 = self.decoder4(decoder4)

        decoder3 = self.up3(decoder4)
        decoder3 = torch.cat(
            [decoder3, encoder3],
            dim=1,
        )
        decoder3 = self.decoder3(decoder3)

        decoder2 = self.up2(decoder3)
        decoder2 = torch.cat(
            [decoder2, encoder2],
            dim=1,
        )
        decoder2 = self.decoder2(decoder2)

        decoder1 = self.up1(decoder2)
        decoder1 = torch.cat(
            [decoder1, encoder1],
            dim=1,
        )
        decoder1 = self.decoder1(decoder1)

        return self.output_layer(decoder1)


def load_model(
    model_path: Path,
    device: torch.device,
) -> tuple[nn.Module, int, int]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The model checkpoint must be a dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    image_height = int(
        checkpoint.get("image_height", 256)
    )
    image_width = int(
        checkpoint.get("image_width", 192)
    )
    base_channels = int(
        checkpoint.get("base_channels", 16)
    )

    model = UNet(
        input_channels=3,
        output_channels=1,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model, image_height, image_width


def load_calibration(
    calibration_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found:\n{calibration_path}"
        )

    calibration = np.load(calibration_path)

    camera_matrix = None
    distortion_coefficients = None

    for key in (
        "camera_matrix",
        "mtx",
        "cameraMatrix",
    ):
        if key in calibration:
            camera_matrix = calibration[key]
            break

    for key in (
        "dist_coeffs",
        "dist",
        "distortion_coefficients",
        "distCoeffs",
    ):
        if key in calibration:
            distortion_coefficients = calibration[key]
            break

    if camera_matrix is None:
        raise KeyError(
            "Camera matrix was not found. "
            f"Available keys: {list(calibration.keys())}"
        )

    if distortion_coefficients is None:
        raise KeyError(
            "Distortion coefficients were not found. "
            f"Available keys: {list(calibration.keys())}"
        )

    return (
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(
            distortion_coefficients,
            dtype=np.float64,
        ),
    )


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]

    new_camera_matrix, _ = (
        cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion_coefficients,
            (width, height),
            alpha=0,
            newImgSize=(width, height),
        )
    )

    return cv2.undistort(
        image,
        camera_matrix,
        distortion_coefficients,
        None,
        new_camera_matrix,
    )


def prepare_image(
    image_bgr: np.ndarray,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    image_pil = Image.fromarray(image_rgb)

    resized = TF.resize(
        image_pil,
        [image_height, image_width],
        interpolation=TF.InterpolationMode.BILINEAR,
    )

    tensor = TF.to_tensor(resized)

    tensor = TF.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    return tensor.unsqueeze(0)


def predict_mask(
    model: nn.Module,
    image_tensor: torch.Tensor,
    output_height: int,
    output_width: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = model(image_tensor)

        probability_map = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

    probability_map = cv2.resize(
        probability_map,
        (output_width, output_height),
        interpolation=cv2.INTER_LINEAR,
    )

    binary_mask = (
        probability_map >= threshold
    ).astype(np.uint8) * 255

    return probability_map, binary_mask


def keep_largest_component(
    binary_mask: np.ndarray,
) -> np.ndarray:
    number_of_labels, labels, statistics, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    if number_of_labels <= 1:
        return binary_mask

    component_areas = statistics[
        1:,
        cv2.CC_STAT_AREA,
    ]

    largest_label = (
        int(np.argmax(component_areas)) + 1
    )

    cleaned_mask = np.zeros_like(binary_mask)
    cleaned_mask[labels == largest_label] = 255

    return cleaned_mask


def get_aruco_detector() -> Any:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV ArUco support is unavailable. Install:\n"
            "pip install opencv-contrib-python"
        )

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = (
            cv2.aruco.DetectorParameters_create()
        )

    # These settings improve detection in mildly blurred or uneven images.
    if hasattr(parameters, "cornerRefinementMethod"):
        parameters.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_SUBPIX
        )

    if hasattr(parameters, "adaptiveThreshWinSizeMin"):
        parameters.adaptiveThreshWinSizeMin = 3

    if hasattr(parameters, "adaptiveThreshWinSizeMax"):
        parameters.adaptiveThreshWinSizeMax = 53

    if hasattr(parameters, "adaptiveThreshWinSizeStep"):
        parameters.adaptiveThreshWinSizeStep = 4

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(
            dictionary,
            parameters,
        )

    return dictionary, parameters


def detect_marker(
    image: np.ndarray,
    marker_id: int,
) -> np.ndarray:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    detector = get_aruco_detector()

    if hasattr(detector, "detectMarkers"):
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        dictionary, parameters = detector

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters,
        )

    if ids is None or len(ids) == 0:
        raise RuntimeError(
            "No ArUco marker was detected. Ensure the full marker "
            "is visible, sharp, and not covered by glare or shadow."
        )

    flattened_ids = ids.flatten()

    matching_indices = np.where(
        flattened_ids == marker_id
    )[0]

    if len(matching_indices) == 0:
        detected = ", ".join(
            str(int(value))
            for value in flattened_ids
        )

        raise RuntimeError(
            f"ArUco marker ID {marker_id} was not found. "
            f"Detected IDs: {detected}"
        )

    marker_index = int(matching_indices[0])

    # ArUco returns corners in this order:
    # top-left, top-right, bottom-right, bottom-left.
    return corners[marker_index].reshape(
        4,
        2,
    ).astype(np.float32)


def find_object_contour(
    binary_mask: np.ndarray,
) -> np.ndarray:
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise RuntimeError(
            "No object contour was found in the predicted mask."
        )

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    if cv2.contourArea(contour) <= 0:
        raise RuntimeError(
            "The detected object contour has zero area."
        )

    return contour.astype(np.float32)


def build_image_to_metric_homography(
    marker_corners: np.ndarray,
    marker_size_mm: float,
) -> np.ndarray:
    metric_marker_corners = np.array(
        [
            [0.0, 0.0],
            [marker_size_mm, 0.0],
            [marker_size_mm, marker_size_mm],
            [0.0, marker_size_mm],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(
        marker_corners.astype(np.float32),
        metric_marker_corners,
    )

    if not np.isfinite(homography).all():
        raise RuntimeError(
            "The marker homography contains invalid values."
        )

    return homography


def transform_contour_to_metric_plane(
    contour: np.ndarray,
    image_to_metric_homography: np.ndarray,
) -> np.ndarray:
    contour_points = contour.reshape(
        -1,
        1,
        2,
    ).astype(np.float32)

    metric_points = cv2.perspectiveTransform(
        contour_points,
        image_to_metric_homography,
    )

    if not np.isfinite(metric_points).all():
        raise RuntimeError(
            "The transformed object contour contains invalid values."
        )

    return metric_points


def measure_metric_contour(
    metric_contour: np.ndarray,
) -> dict[str, Any]:
    rectangle = cv2.minAreaRect(
        metric_contour
    )

    center = rectangle[0]
    width_mm = float(rectangle[1][0])
    height_mm = float(rectangle[1][1])
    angle = float(rectangle[2])

    if width_mm <= 0 or height_mm <= 0:
        raise RuntimeError(
            "The metric object rectangle is invalid."
        )

    long_side_mm = max(
        width_mm,
        height_mm,
    )
    short_side_mm = min(
        width_mm,
        height_mm,
    )

    metric_box_points = cv2.boxPoints(
        rectangle
    ).astype(np.float32)

    metric_area_mm2 = float(
        cv2.contourArea(metric_contour)
    )

    return {
        "center_metric_mm": [
            float(center[0]),
            float(center[1]),
        ],
        "angle_degrees": angle,
        "long_side_mm": long_side_mm,
        "short_side_mm": short_side_mm,
        "metric_box_points": metric_box_points,
        "contour_area_mm2": metric_area_mm2,
    }


def metric_box_to_image(
    metric_box_points: np.ndarray,
    image_to_metric_homography: np.ndarray,
) -> np.ndarray:
    metric_to_image_homography = np.linalg.inv(
        image_to_metric_homography
    )

    image_points = cv2.perspectiveTransform(
        metric_box_points.reshape(-1, 1, 2),
        metric_to_image_homography,
    )

    return np.round(
        image_points.reshape(-1, 2)
    ).astype(np.int32)


def calculate_marker_diagnostics(
    marker_corners: np.ndarray,
    marker_size_mm: float,
) -> dict[str, Any]:
    top_left, top_right, bottom_right, bottom_left = (
        marker_corners
    )

    side_lengths_pixels = [
        float(np.linalg.norm(top_right - top_left)),
        float(np.linalg.norm(bottom_right - top_right)),
        float(np.linalg.norm(bottom_left - bottom_right)),
        float(np.linalg.norm(top_left - bottom_left)),
    ]

    average_side_pixels = float(
        np.mean(side_lengths_pixels)
    )

    approximate_pixels_per_mm = (
        average_side_pixels / marker_size_mm
    )

    return {
        "side_lengths_pixels": side_lengths_pixels,
        "average_side_pixels": average_side_pixels,
        "approximate_pixels_per_mm": (
            approximate_pixels_per_mm
        ),
    }


def draw_result(
    image: np.ndarray,
    binary_mask: np.ndarray,
    probability_map: np.ndarray,
    marker_corners: np.ndarray,
    marker_id: int,
    marker_size_mm: float,
    image_box_points: np.ndarray,
    measurement: dict[str, Any],
) -> tuple[np.ndarray, float]:
    output = image.copy()

    mask_pixels = binary_mask > 0

    if mask_pixels.any():
        color_layer = np.zeros_like(output)
        color_layer[:] = (0, 0, 255)

        blended = cv2.addWeighted(
            output,
            0.55,
            color_layer,
            0.45,
            0,
        )

        output[mask_pixels] = blended[mask_pixels]

        confidence = float(
            probability_map[mask_pixels].mean()
        )
    else:
        confidence = 0.0

    cv2.polylines(
        output,
        [image_box_points],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    marker_points = np.round(
        marker_corners
    ).astype(np.int32)

    cv2.polylines(
        output,
        [marker_points],
        isClosed=True,
        color=(255, 0, 255),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    marker_center = marker_corners.mean(
        axis=0
    ).astype(int)

    cv2.putText(
        output,
        (
            f"ArUco ID {marker_id}: "
            f"{marker_size_mm:.1f} mm"
        ),
        (
            int(marker_center[0]) + 10,
            int(marker_center[1]) - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    text_lines = [
        (
            f"Long side: "
            f"{measurement['long_side_mm']:.2f} mm"
        ),
        (
            f"Short side: "
            f"{measurement['short_side_mm']:.2f} mm"
        ),
        (
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        ),
        "Method: planar ArUco homography",
    ]

    panel_width = 570
    panel_height = 45 + 38 * len(text_lines)

    cv2.rectangle(
        output,
        (10, 10),
        (panel_width, panel_height),
        (0, 0, 0),
        thickness=-1,
    )

    for index, line in enumerate(text_lines):
        cv2.putText(
            output,
            line,
            (25, 48 + index * 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output, confidence


def save_json(
    output_path: Path,
    data: dict[str, Any],
) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, (np.floating, np.integer)):
            return value.item()

        if isinstance(value, dict):
            return {
                key: convert(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                convert(item)
                for item in value
            ]

        return value

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            convert(data),
            file,
            indent=2,
        )


def run_measurement(
    image_path: Path,
    model_path: Path,
    calibration_path: Path,
    output_directory: Path,
    marker_id: int,
    marker_size_mm: float,
    threshold: float,
    already_undistorted: bool,
) -> None:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{image_path}"
        )

    if (
        image_path.suffix.lower()
        not in SUPPORTED_IMAGE_EXTENSIONS
    ):
        raise ValueError(
            f"Unsupported image type: {image_path.suffix}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"OpenCV could not read:\n{image_path}"
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

    if already_undistorted:
        working_image = image
    else:
        camera_matrix, distortion_coefficients = (
            load_calibration(
                calibration_path
            )
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

    image_contour = find_object_contour(
        binary_mask
    )

    metric_contour = (
        transform_contour_to_metric_plane(
            image_contour,
            image_to_metric_homography,
        )
    )

    measurement = measure_metric_contour(
        metric_contour
    )

    image_box_points = metric_box_to_image(
        measurement["metric_box_points"],
        image_to_metric_homography,
    )

    marker_diagnostics = (
        calculate_marker_diagnostics(
            marker_corners,
            marker_size_mm,
        )
    )

    annotated_image, confidence = draw_result(
        image=working_image,
        binary_mask=binary_mask,
        probability_map=probability_map,
        marker_corners=marker_corners,
        marker_id=marker_id,
        marker_size_mm=marker_size_mm,
        image_box_points=image_box_points,
        measurement=measurement,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = image_path.stem

    processed_path = (
        output_directory
        / f"{stem}_processed.jpg"
    )
    mask_path = (
        output_directory
        / f"{stem}_mask.png"
    )
    annotated_path = (
        output_directory
        / f"{stem}_measurement.jpg"
    )
    result_path = (
        output_directory
        / f"{stem}_measurement.json"
    )

    saved_images = {
        processed_path: working_image,
        mask_path: binary_mask,
        annotated_path: annotated_image,
    }

    for save_path, output_image in saved_images.items():
        if not cv2.imwrite(
            str(save_path),
            output_image,
        ):
            raise RuntimeError(
                f"Failed to save output:\n{save_path}"
            )

    result = {
        "input_image": str(image_path),
        "model_path": str(model_path),
        "calibration_path": (
            None
            if already_undistorted
            else str(calibration_path)
        ),
        "already_undistorted": (
            already_undistorted
        ),
        "device": str(device),
        "threshold": threshold,
        "measurement_method": (
            "planar_homography_from_aruco_marker"
        ),
        "marker_dictionary": "DICT_4X4_50",
        "marker_id": marker_id,
        "marker_size_mm": marker_size_mm,
        "marker_corners_pixels": marker_corners,
        "marker_diagnostics": marker_diagnostics,
        "image_to_metric_homography": (
            image_to_metric_homography
        ),
        "long_side_mm": measurement[
            "long_side_mm"
        ],
        "short_side_mm": measurement[
            "short_side_mm"
        ],
        "contour_area_mm2": measurement[
            "contour_area_mm2"
        ],
        "object_center_metric_mm": measurement[
            "center_metric_mm"
        ],
        "object_angle_degrees": measurement[
            "angle_degrees"
        ],
        "metric_box_points_mm": measurement[
            "metric_box_points"
        ],
        "image_box_points_pixels": (
            image_box_points
        ),
        "mean_foreground_confidence": confidence,
        "output_image": str(annotated_path),
        "output_mask": str(mask_path),
    }

    save_json(
        result_path,
        result,
    )

    print("========================================")
    print("OBJECT MEASUREMENT COMPLETE")
    print("========================================")
    print(f"Device:              {device}")
    print(f"Input image:         {image_path}")
    print(f"Marker ID:           {marker_id}")
    print(
        f"Marker size:         "
        f"{marker_size_mm:.2f} mm"
    )
    print(
        "Method:              "
        "planar ArUco homography"
    )
    print(
        f"Long side:           "
        f"{measurement['long_side_mm']:.2f} mm"
    )
    print(
        f"Short side:          "
        f"{measurement['short_side_mm']:.2f} mm"
    )
    print(
        f"Mean confidence:     "
        f"{confidence * 100:.2f}%"
    )
    print()
    print("Important:")
    print(
        "  The marker and measured object surface must be "
        "on the same physical plane."
    )
    print()
    print("Saved outputs:")
    print(f"  {processed_path}")
    print(f"  {mask_path}")
    print(f"  {annotated_path}")
    print(f"  {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a segmented planar object using an "
            "ArUco-marker homography."
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help=(
            "Path to an image containing the object "
            "and ArUco marker."
        ),
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
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for measurement outputs.",
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
        help=(
            "Physical outer side length of the printed "
            "ArUco marker in millimetres."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Segmentation probability threshold.",
    )

    parser.add_argument(
        "--already-undistorted",
        action="store_true",
        help=(
            "Skip camera undistortion when the input "
            "image has already been undistorted."
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

    run_measurement(
        image_path=args.image,
        model_path=args.model,
        calibration_path=args.calibration,
        output_directory=args.output_directory,
        marker_id=args.marker_id,
        marker_size_mm=args.marker_size_mm,
        threshold=args.threshold,
        already_undistorted=args.already_undistorted,
    )


if __name__ == "__main__":
    main()