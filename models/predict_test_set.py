from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGE_DIRECTORY = PROJECT_ROOT / "dataset" / "coco_split" / "test" / "images"
TEST_MASK_DIRECTORY = PROJECT_ROOT / "dataset" / "coco_split" / "test" / "masks"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "training" / "outputs" / "best_unet.pt"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "evaluation" / "outputs" / "test_predictions"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UNet(nn.Module):
    def __init__(self, input_channels: int = 3, output_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        self.encoder1 = DoubleConv(input_channels, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels * 2)
        self.encoder3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.encoder4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(base_channels * 2, base_channels)
        self.output_layer = nn.Conv2d(base_channels, output_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        bottleneck = self.bottleneck(self.pool(encoder4))

        decoder4 = self.up4(bottleneck)
        decoder4 = self.decoder4(torch.cat([decoder4, encoder4], dim=1))
        decoder3 = self.up3(decoder4)
        decoder3 = self.decoder3(torch.cat([decoder3, encoder3], dim=1))
        decoder2 = self.up2(decoder3)
        decoder2 = self.decoder2(torch.cat([decoder2, encoder2], dim=1))
        decoder1 = self.up1(decoder2)
        decoder1 = self.decoder1(torch.cat([decoder1, encoder1], dim=1))
        return self.output_layer(decoder1)


def load_model(model_path: Path, device: torch.device) -> tuple[nn.Module, int, int]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found:\n{model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    image_height = int(checkpoint.get("image_height", 256))
    image_width = int(checkpoint.get("image_width", 192))
    base_channels = int(checkpoint.get("base_channels", 16))

    model = UNet(input_channels=3, output_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, image_height, image_width


def prepare_image(image_bgr: np.ndarray, image_height: int, image_width: int) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    resized = TF.resize(
        image_pil,
        [image_height, image_width],
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0)


def predict_mask(
    model: nn.Module,
    image_tensor: torch.Tensor,
    original_height: int,
    original_width: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = model(image_tensor)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

    probability_full = cv2.resize(
        probability,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    binary_mask = (probability_full >= threshold).astype(np.uint8) * 255
    return probability_full, binary_mask


def calculate_metrics(prediction_mask: np.ndarray, ground_truth_mask: np.ndarray) -> tuple[float, float]:
    prediction = prediction_mask > 0
    ground_truth = ground_truth_mask > 0
    intersection = np.logical_and(prediction, ground_truth).sum()
    union = np.logical_or(prediction, ground_truth).sum()
    prediction_pixels = prediction.sum()
    ground_truth_pixels = ground_truth.sum()
    iou = float(intersection / union) if union > 0 else 1.0
    denominator = prediction_pixels + ground_truth_pixels
    dice = float(2.0 * intersection / denominator) if denominator > 0 else 1.0
    return iou, dice


def create_mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    mask_pixels = mask > 0
    highlighted = overlay.copy()
    highlighted[mask_pixels] = (
        highlighted[mask_pixels] * 0.45 + np.array([0, 0, 255]) * 0.55
    ).astype(np.uint8)
    overlay[mask_pixels] = highlighted[mask_pixels]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    return overlay


def create_comparison(
    original: np.ndarray,
    ground_truth_overlay: np.ndarray,
    prediction_overlay: np.ndarray,
    iou: float,
    dice: float,
    confidence: float,
) -> np.ndarray:
    height, _ = original.shape[:2]
    original_panel = original.copy()

    cv2.putText(original_panel, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(ground_truth_overlay, "Ground Truth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(prediction_overlay, "Prediction", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    metric_text = f"IoU: {iou:.4f}  Dice: {dice:.4f}  Confidence: {confidence:.4f}"
    cv2.putText(
        prediction_overlay,
        metric_text,
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return np.hstack([original_panel, ground_truth_overlay, prediction_overlay])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate U-Net prediction overlays for the held-out test set."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if not TEST_IMAGE_DIRECTORY.exists():
        raise FileNotFoundError(f"Test image directory not found:\n{TEST_IMAGE_DIRECTORY}")
    if not TEST_MASK_DIRECTORY.exists():
        raise FileNotFoundError(f"Test mask directory not found:\n{TEST_MASK_DIRECTORY}")

    image_paths = sorted(
        path
        for path in TEST_IMAGE_DIRECTORY.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError("No test images were found.")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_height, image_width = load_model(args.model, device)

    print("========================================")
    print("TEST-SET PREDICTION VISUALIZATION")
    print("========================================")
    print(f"Device:          {device}")
    print(f"Model:           {args.model}")
    print(f"Images:          {len(image_paths)}")
    print(f"Model resolution:{image_width:>5} x {image_height}")
    print(f"Threshold:       {args.threshold}")
    print()

    for image_path in tqdm(image_paths, desc="Generating predictions"):
        mask_path = TEST_MASK_DIRECTORY / f"{image_path.stem}.png"
        if not mask_path.exists():
            print(f"\n[SKIPPED] Missing mask: {mask_path.name}")
            continue

        image = cv2.imread(str(image_path))
        ground_truth_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"\n[SKIPPED] Could not read image: {image_path.name}")
            continue
        if ground_truth_mask is None:
            print(f"\n[SKIPPED] Could not read mask: {mask_path.name}")
            continue

        original_height, original_width = image.shape[:2]
        if ground_truth_mask.shape != (original_height, original_width):
            ground_truth_mask = cv2.resize(
                ground_truth_mask,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )

        image_tensor = prepare_image(image, image_height, image_width).to(device)
        probability_map, prediction_mask = predict_mask(
            model,
            image_tensor,
            original_height,
            original_width,
            args.threshold,
        )

        iou, dice = calculate_metrics(prediction_mask, ground_truth_mask)
        predicted_pixels = prediction_mask > 0
        confidence = float(probability_map[predicted_pixels].mean()) if predicted_pixels.any() else 0.0

        ground_truth_overlay = create_mask_overlay(image, ground_truth_mask)
        prediction_overlay = create_mask_overlay(image, prediction_mask)
        comparison = create_comparison(
            image,
            ground_truth_overlay,
            prediction_overlay,
            iou,
            dice,
            confidence,
        )

        cv2.imwrite(str(args.output_directory / f"{image_path.stem}_predicted_mask.png"), prediction_mask)
        cv2.imwrite(str(args.output_directory / f"{image_path.stem}_overlay.jpg"), prediction_overlay)
        cv2.imwrite(str(args.output_directory / f"{image_path.stem}_comparison.jpg"), comparison)

    print()
    print("========================================")
    print("PREDICTIONS COMPLETE")
    print("========================================")
    print(f"Output directory:\n{args.output_directory}")


if __name__ == "__main__":
    main()