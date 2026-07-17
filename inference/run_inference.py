from __future__ import annotations

import argparse
from pathlib import Path

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
    / "inference"
    / "outputs"
)

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

    possible_camera_keys = (
        "camera_matrix",
        "mtx",
        "cameraMatrix",
    )

    possible_distortion_keys = (
        "dist_coeffs",
        "dist",
        "distortion_coefficients",
        "distCoeffs",
    )

    camera_matrix = None
    distortion_coefficients = None

    for key in possible_camera_keys:
        if key in calibration:
            camera_matrix = calibration[key]
            break

    for key in possible_distortion_keys:
        if key in calibration:
            distortion_coefficients = calibration[key]
            break

    if camera_matrix is None:
        raise KeyError(
            "Camera matrix was not found in the calibration file. "
            f"Available keys: {list(calibration.keys())}"
        )

    if distortion_coefficients is None:
        raise KeyError(
            "Distortion coefficients were not found in the calibration file. "
            f"Available keys: {list(calibration.keys())}"
        )

    return (
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion_coefficients, dtype=np.float64),
    )


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    crop: bool,
) -> np.ndarray:
    height, width = image.shape[:2]

    new_camera_matrix, valid_roi = (
        cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion_coefficients,
            (width, height),
            alpha=0,
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

    if crop:
        x, y, roi_width, roi_height = valid_roi

        if roi_width > 0 and roi_height > 0:
            undistorted = undistorted[
                y : y + roi_height,
                x : x + roi_width,
            ]

    return undistorted


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

    resized_image = TF.resize(
        image_pil,
        [image_height, image_width],
        interpolation=TF.InterpolationMode.BILINEAR,
    )

    image_tensor = TF.to_tensor(
        resized_image
    )

    image_tensor = TF.normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    return image_tensor.unsqueeze(0)


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

    largest_component_label = (
        1 + int(np.argmax(component_areas))
    )

    cleaned_mask = np.zeros_like(
        binary_mask
    )

    cleaned_mask[
        labels == largest_component_label
    ] = 255

    return cleaned_mask


def create_overlay(
    image: np.ndarray,
    binary_mask: np.ndarray,
    probability_map: np.ndarray,
) -> tuple[np.ndarray, float]:
    overlay = image.copy()

    mask_pixels = binary_mask > 0

    if mask_pixels.any():
        colored_layer = np.zeros_like(image)
        colored_layer[:, :] = (0, 0, 255)

        overlay[mask_pixels] = cv2.addWeighted(
            image,
            0.45,
            colored_layer,
            0.55,
            0,
        )[mask_pixels]

        mean_confidence = float(
            probability_map[mask_pixels].mean()
        )
    else:
        mean_confidence = 0.0

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=3,
    )

    label = (
        f"Object confidence: "
        f"{mean_confidence * 100:.2f}%"
    )

    cv2.rectangle(
        overlay,
        (10, 10),
        (420, 55),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        overlay,
        label,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return overlay, mean_confidence


def create_probability_visualization(
    probability_map: np.ndarray,
) -> np.ndarray:
    normalized = np.clip(
        probability_map * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET,
    )


def run_inference(
    image_path: Path,
    model_path: Path,
    calibration_path: Path,
    output_directory: Path,
    threshold: float,
    crop_undistorted: bool,
    keep_largest: bool,
) -> None:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{image_path}"
        )

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image type: {image_path.suffix}"
        )

    original_image = cv2.imread(
        str(image_path)
    )

    if original_image is None:
        raise RuntimeError(
            f"OpenCV could not read the image:\n{image_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, model_height, model_width = (
        load_model(
            model_path=model_path,
            device=device,
        )
    )

    camera_matrix, distortion_coefficients = (
        load_calibration(
            calibration_path=calibration_path,
        )
    )

    undistorted_image = undistort_image(
        image=original_image,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        crop=crop_undistorted,
    )

    output_height, output_width = (
        undistorted_image.shape[:2]
    )

    image_tensor = prepare_image(
        image_bgr=undistorted_image,
        image_height=model_height,
        image_width=model_width,
    ).to(device)

    probability_map, binary_mask = (
        predict_mask(
            model=model,
            image_tensor=image_tensor,
            output_height=output_height,
            output_width=output_width,
            threshold=threshold,
        )
    )

    if keep_largest:
        binary_mask = keep_largest_component(
            binary_mask
        )

    overlay, mean_confidence = create_overlay(
        image=undistorted_image,
        binary_mask=binary_mask,
        probability_map=probability_map,
    )

    probability_visualization = (
        create_probability_visualization(
            probability_map
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = image_path.stem

    undistorted_path = (
        output_directory
        / f"{stem}_undistorted.jpg"
    )

    mask_path = (
        output_directory
        / f"{stem}_predicted_mask.png"
    )

    overlay_path = (
        output_directory
        / f"{stem}_overlay.jpg"
    )

    probability_path = (
        output_directory
        / f"{stem}_probability.jpg"
    )

    comparison_path = (
        output_directory
        / f"{stem}_comparison.jpg"
    )

    comparison = np.hstack(
        [
            undistorted_image,
            overlay,
        ]
    )

    cv2.putText(
        comparison,
        "Undistorted input",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        comparison,
        "Segmentation prediction",
        (output_width + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    saved_results = {
        undistorted_path: undistorted_image,
        mask_path: binary_mask,
        overlay_path: overlay,
        probability_path: probability_visualization,
        comparison_path: comparison,
    }

    for save_path, output_image in saved_results.items():
        success = cv2.imwrite(
            str(save_path),
            output_image,
        )

        if not success:
            raise RuntimeError(
                f"Failed to save output:\n{save_path}"
            )

    foreground_pixels = int(
        np.count_nonzero(binary_mask)
    )

    total_pixels = int(
        binary_mask.size
    )

    foreground_percentage = (
        100.0
        * foreground_pixels
        / total_pixels
    )

    print("========================================")
    print("U-NET INFERENCE COMPLETE")
    print("========================================")
    print(f"Device:               {device}")
    print(f"Input image:          {image_path}")
    print(f"Model:                {model_path}")
    print(f"Calibration:          {calibration_path}")
    print(
        f"Model input size:     "
        f"{model_width} x {model_height}"
    )
    print(
        f"Output image size:    "
        f"{output_width} x {output_height}"
    )
    print(f"Threshold:            {threshold:.2f}")
    print(
        f"Mean confidence:      "
        f"{mean_confidence * 100:.2f}%"
    )
    print(
        f"Foreground coverage:  "
        f"{foreground_percentage:.2f}%"
    )
    print()
    print("Saved outputs:")
    print(f"  {undistorted_path}")
    print(f"  {mask_path}")
    print(f"  {overlay_path}")
    print(f"  {probability_path}")
    print(f"  {comparison_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Undistort an input image and run "
            "U-Net object-segmentation inference."
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image.",
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
        help="Directory where inference outputs are saved.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold used to create "
            "the binary segmentation mask."
        ),
    )

    parser.add_argument(
        "--crop-undistorted",
        action="store_true",
        help=(
            "Crop the undistorted image to OpenCV's "
            "valid region of interest."
        ),
    )

    parser.add_argument(
        "--keep-all-components",
        action="store_true",
        help=(
            "Keep every predicted connected component. "
            "By default only the largest component is retained."
        ),
    )

    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    run_inference(
        image_path=args.image,
        model_path=args.model,
        calibration_path=args.calibration,
        output_directory=args.output_directory,
        threshold=args.threshold,
        crop_undistorted=args.crop_undistorted,
        keep_largest=not args.keep_all_components,
    )


if __name__ == "__main__":
    main()