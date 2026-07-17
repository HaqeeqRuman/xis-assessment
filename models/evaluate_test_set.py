from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "coco_split"
)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "outputs"
    / "best_unet.pt"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "evaluation"
    / "outputs"
)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


class BoxSegmentationDataset(Dataset):
    def __init__(
        self,
        split: str,
        image_height: int,
        image_width: int,
    ) -> None:
        self.split = split
        self.image_height = image_height
        self.image_width = image_width

        self.image_directory = (
            DATASET_ROOT
            / split
            / "images"
        )

        self.mask_directory = (
            DATASET_ROOT
            / split
            / "masks"
        )

        if not self.image_directory.exists():
            raise FileNotFoundError(
                f"Image directory not found:\n"
                f"{self.image_directory}"
            )

        if not self.mask_directory.exists():
            raise FileNotFoundError(
                f"Mask directory not found:\n"
                f"{self.mask_directory}"
            )

        self.image_paths = sorted(
            path
            for path in self.image_directory.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No test images found in:\n"
                f"{self.image_directory}"
            )

        missing_masks: list[str] = []

        for image_path in self.image_paths:
            mask_path = (
                self.mask_directory
                / f"{image_path.stem}.png"
            )

            if not mask_path.exists():
                missing_masks.append(mask_path.name)

        if missing_masks:
            preview = "\n".join(
                f"  - {file_name}"
                for file_name in missing_masks[:20]
            )

            raise FileNotFoundError(
                f"{len(missing_masks)} masks are missing:\n"
                f"{preview}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.image_paths[index]

        mask_path = (
            self.mask_directory
            / f"{image_path.stem}.png"
        )

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(
            image,
            [
                self.image_height,
                self.image_width,
            ],
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        mask = TF.resize(
            mask,
            [
                self.image_height,
                self.image_width,
            ],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        image_tensor = TF.to_tensor(image)

        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask_array = np.asarray(
            mask,
            dtype=np.float32,
        )

        mask_array = (
            mask_array > 127
        ).astype(np.float32)

        mask_tensor = torch.from_numpy(
            mask_array
        ).unsqueeze(0)

        return (
            image_tensor,
            mask_tensor,
            image_path.name,
        )


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


def safe_divide(
    numerator: float,
    denominator: float,
    epsilon: float = 1e-7,
) -> float:
    return float(
        numerator
        / (denominator + epsilon)
    )


def calculate_binary_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)

    true_positive = float(
        (prediction * target).sum().item()
    )

    false_positive = float(
        (
            prediction
            * (1.0 - target)
        ).sum().item()
    )

    false_negative = float(
        (
            (1.0 - prediction)
            * target
        ).sum().item()
    )

    true_negative = float(
        (
            (1.0 - prediction)
            * (1.0 - target)
        ).sum().item()
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    f1_score = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    iou = safe_divide(
        true_positive,
        true_positive
        + false_positive
        + false_negative,
    )

    dice = safe_divide(
        2.0 * true_positive,
        2.0 * true_positive
        + false_positive
        + false_negative,
    )

    accuracy = safe_divide(
        true_positive + true_negative,
        true_positive
        + true_negative
        + false_positive
        + false_negative,
    )

    specificity = safe_divide(
        true_negative,
        true_negative + false_positive,
    )

    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "specificity": specificity,
    }


def load_checkpoint(
    model_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n"
            f"{model_path}"
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
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    return checkpoint


def save_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not rows:
        raise ValueError(
            "No metric rows were generated."
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained U-Net on the "
            "held-out segmentation test set."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
            "test",
        ],
        default="test",
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold used to convert "
            "predictions into binary masks."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = load_checkpoint(
        model_path=args.model,
        device=device,
    )

    image_height = int(
        checkpoint.get(
            "image_height",
            256,
        )
    )

    image_width = int(
        checkpoint.get(
            "image_width",
            192,
        )
    )

    base_channels = int(
        checkpoint.get(
            "base_channels",
            16,
        )
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

    dataset = BoxSegmentationDataset(
        split=args.split,
        image_height=image_height,
        image_width=image_width,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("========================================")
    print("U-NET SEGMENTATION EVALUATION")
    print("========================================")
    print(f"Device:          {device}")
    print(f"Model:           {args.model}")
    print(f"Dataset split:   {args.split}")
    print(f"Number of images:{len(dataset):>4}")
    print(
        f"Input resolution: "
        f"{image_width} x {image_height}"
    )
    print(f"Threshold:       {args.threshold}")
    print()

    per_image_rows: list[dict[str, Any]] = []

    total_true_positive = 0.0
    total_true_negative = 0.0
    total_false_positive = 0.0
    total_false_negative = 0.0

    with torch.no_grad():
        for (
            images,
            masks,
            file_names,
        ) in tqdm(
            data_loader,
            desc="Evaluating",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities
                >= args.threshold
            ).float()

            for index in range(images.size(0)):
                metrics = calculate_binary_metrics(
                    prediction=predictions[index],
                    target=masks[index],
                )

                total_true_positive += (
                    metrics["true_positive"]
                )

                total_true_negative += (
                    metrics["true_negative"]
                )

                total_false_positive += (
                    metrics["false_positive"]
                )

                total_false_negative += (
                    metrics["false_negative"]
                )

                predicted_probability = (
                    probabilities[index]
                )

                predicted_pixels = (
                    predictions[index] > 0
                )

                if predicted_pixels.any():
                    mean_foreground_confidence = float(
                        predicted_probability[
                            predicted_pixels
                        ].mean().item()
                    )
                else:
                    mean_foreground_confidence = 0.0

                per_image_rows.append(
                    {
                        "file_name": file_names[index],
                        "iou": metrics["iou"],
                        "dice": metrics["dice"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                        "accuracy": metrics["accuracy"],
                        "specificity": metrics[
                            "specificity"
                        ],
                        "mean_foreground_confidence": (
                            mean_foreground_confidence
                        ),
                    }
                )

    global_metrics = calculate_binary_metrics(
        prediction=torch.tensor(
            [
                1.0,
            ]
            * int(total_true_positive)
            + [
                1.0,
            ]
            * int(total_false_positive)
            + [
                0.0,
            ]
            * int(total_false_negative)
            + [
                0.0,
            ]
            * int(total_true_negative)
        ),
        target=torch.tensor(
            [
                1.0,
            ]
            * int(total_true_positive)
            + [
                0.0,
            ]
            * int(total_false_positive)
            + [
                1.0,
            ]
            * int(total_false_negative)
            + [
                0.0,
            ]
            * int(total_true_negative)
        ),
    )

    mean_metrics = {
        metric_name: float(
            np.mean(
                [
                    row[metric_name]
                    for row in per_image_rows
                ]
            )
        )
        for metric_name in [
            "iou",
            "dice",
            "precision",
            "recall",
            "f1_score",
            "accuracy",
            "specificity",
            "mean_foreground_confidence",
        ]
    }

    global_precision = safe_divide(
        total_true_positive,
        total_true_positive
        + total_false_positive,
    )

    global_recall = safe_divide(
        total_true_positive,
        total_true_positive
        + total_false_negative,
    )

    global_f1 = safe_divide(
        2.0
        * global_precision
        * global_recall,
        global_precision + global_recall,
    )

    global_iou = safe_divide(
        total_true_positive,
        total_true_positive
        + total_false_positive
        + total_false_negative,
    )

    global_dice = safe_divide(
        2.0 * total_true_positive,
        2.0 * total_true_positive
        + total_false_positive
        + total_false_negative,
    )

    global_accuracy = safe_divide(
        total_true_positive
        + total_true_negative,
        total_true_positive
        + total_true_negative
        + total_false_positive
        + total_false_negative,
    )

    global_specificity = safe_divide(
        total_true_negative,
        total_true_negative
        + total_false_positive,
    )

    results = {
        "model_path": str(args.model),
        "dataset_split": args.split,
        "number_of_images": len(dataset),
        "device": str(device),
        "threshold": args.threshold,
        "input_width": image_width,
        "input_height": image_height,
        "checkpoint_epoch": checkpoint.get(
            "epoch"
        ),
        "checkpoint_validation_dice": (
            checkpoint.get(
                "validation_dice"
            )
        ),
        "checkpoint_validation_iou": (
            checkpoint.get(
                "validation_iou"
            )
        ),
        "mean_per_image_metrics": mean_metrics,
        "global_pixel_metrics": {
            "iou": global_iou,
            "dice": global_dice,
            "precision": global_precision,
            "recall": global_recall,
            "f1_score": global_f1,
            "accuracy": global_accuracy,
            "specificity": global_specificity,
        },
        "confusion_matrix_pixels": {
            "true_positive": total_true_positive,
            "true_negative": total_true_negative,
            "false_positive": total_false_positive,
            "false_negative": total_false_negative,
        },
    }

    per_image_csv_path = (
        args.output_directory
        / f"{args.split}_per_image_metrics.csv"
    )

    metrics_json_path = (
        args.output_directory
        / f"{args.split}_metrics.json"
    )

    summary_csv_path = (
        args.output_directory
        / f"{args.split}_summary_metrics.csv"
    )

    save_csv(
        rows=per_image_rows,
        output_path=per_image_csv_path,
    )

    save_json(
        data=results,
        output_path=metrics_json_path,
    )

    summary_rows = [
        {
            "metric": metric_name,
            "mean_per_image": mean_metrics[
                metric_name
            ],
            "global_pixel": (
                results[
                    "global_pixel_metrics"
                ].get(
                    metric_name,
                    ""
                )
            ),
        }
        for metric_name in [
            "iou",
            "dice",
            "precision",
            "recall",
            "f1_score",
            "accuracy",
            "specificity",
        ]
    ]

    save_csv(
        rows=summary_rows,
        output_path=summary_csv_path,
    )

    print()
    print("========================================")
    print("TEST RESULTS")
    print("========================================")
    print(
        f"Mean IoU:       "
        f"{mean_metrics['iou']:.4f}"
    )
    print(
        f"Mean Dice:      "
        f"{mean_metrics['dice']:.4f}"
    )
    print(
        f"Mean Precision: "
        f"{mean_metrics['precision']:.4f}"
    )
    print(
        f"Mean Recall:    "
        f"{mean_metrics['recall']:.4f}"
    )
    print(
        f"Mean F1:        "
        f"{mean_metrics['f1_score']:.4f}"
    )
    print(
        f"Mean Accuracy:  "
        f"{mean_metrics['accuracy']:.4f}"
    )
    print(
        f"Mean Confidence:"
        f" {mean_metrics['mean_foreground_confidence']:.4f}"
    )

    print()
    print("Global pixel metrics:")
    print(f"  IoU:       {global_iou:.4f}")
    print(f"  Dice:      {global_dice:.4f}")
    print(f"  Precision: {global_precision:.4f}")
    print(f"  Recall:    {global_recall:.4f}")
    print(f"  F1:        {global_f1:.4f}")
    print(f"  Accuracy:  {global_accuracy:.4f}")

    print()
    print("Saved:")
    print(f"  {per_image_csv_path}")
    print(f"  {summary_csv_path}")
    print(f"  {metrics_json_path}")


if __name__ == "__main__":
    main()