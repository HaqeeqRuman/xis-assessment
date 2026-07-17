from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
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

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "outputs"
)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BoxSegmentationDataset(Dataset):
    def __init__(
        self,
        split: str,
        image_size: tuple[int, int],
        augment: bool = False,
    ) -> None:
        self.split = split
        self.image_size = image_size
        self.augment = augment

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
                f"No images found in:\n"
                f"{self.image_directory}"
            )

        missing_masks = []

        for image_path in self.image_paths:
            mask_path = (
                self.mask_directory
                / f"{image_path.stem}.png"
            )

            if not mask_path.exists():
                missing_masks.append(mask_path.name)

        if missing_masks:
            preview = "\n".join(
                f"  - {name}"
                for name in missing_masks[:20]
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]

        mask_path = (
            self.mask_directory
            / f"{image_path.stem}.png"
        )

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        target_height, target_width = self.image_size

        image = TF.resize(
            image,
            [target_height, target_width],
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        mask = TF.resize(
            mask,
            [target_height, target_width],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < 0.25:
                image = TF.adjust_brightness(
                    image,
                    brightness_factor=random.uniform(
                        0.85,
                        1.15,
                    ),
                )

            if random.random() < 0.25:
                image = TF.adjust_contrast(
                    image,
                    contrast_factor=random.uniform(
                        0.85,
                        1.15,
                    ),
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

        return image_tensor, mask_tensor


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
        base_channels: int = 32,
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


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce_loss = self.bce(
            logits,
            targets,
        )

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.view(
            probabilities.size(0),
            -1,
        )

        targets_flat = targets.view(
            targets.size(0),
            -1,
        )

        intersection = (
            probabilities
            * targets_flat
        ).sum(dim=1)

        dice_score = (
            2.0 * intersection
            + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets_flat.sum(dim=1)
            + self.smooth
        )

        dice_loss = 1.0 - dice_score.mean()

        return bce_loss + dice_loss


def calculate_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> tuple[float, float]:
    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1,
    )

    targets = targets.view(
        targets.size(0),
        -1,
    )

    intersection = (
        predictions
        * targets
    ).sum(dim=1)

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum(dim=1)

    iou = (
        intersection + epsilon
    ) / (
        union + epsilon
    )

    dice = (
        2.0 * intersection + epsilon
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + epsilon
    )

    return (
        float(iou.mean().item()),
        float(dice.mean().item()),
    )


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float, float]:
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_samples = 0

    progress_bar = tqdm(
        data_loader,
        leave=False,
    )

    for images, masks in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        batch_size = images.size(0)

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):
            logits = model(images)

            loss = loss_function(
                logits,
                masks,
            )

            if training:
                loss.backward()
                optimizer.step()

        iou, dice = calculate_metrics(
            logits.detach(),
            masks,
        )

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_iou += iou * batch_size
        total_dice += dice * batch_size
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            iou=f"{iou:.4f}",
            dice=f"{dice:.4f}",
        )

    return (
        total_loss / total_samples,
        total_iou / total_samples,
        total_dice / total_samples,
    )


def save_training_plot(
    history: list[dict[str, float]],
    output_path: Path,
) -> None:
    epochs = [
        row["epoch"]
        for row in history
    ]

    train_loss = [
        row["train_loss"]
        for row in history
    ]

    val_loss = [
        row["val_loss"]
        for row in history
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_loss,
        label="Train loss",
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
    )
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_random_seed(args.seed)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("========================================")
    print("CUSTOM U-NET TRAINING")
    print("========================================")
    print(f"Device: {device}")
    print(
        f"Training resolution: "
        f"{args.width} x {args.height}"
    )
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print()

    train_dataset = BoxSegmentationDataset(
        split="train",
        image_size=(
            args.height,
            args.width,
        ),
        augment=True,
    )

    val_dataset = BoxSegmentationDataset(
        split="val",
        image_size=(
            args.height,
            args.width,
        ),
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(
        input_channels=3,
        output_channels=1,
        base_channels=args.base_channels,
    ).to(device)

    loss_function = DiceBCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
    )

    best_validation_dice = -1.0
    history: list[dict[str, float]] = []

    best_model_path = (
        OUTPUT_DIRECTORY
        / "best_unet.pt"
    )

    latest_model_path = (
        OUTPUT_DIRECTORY
        / "latest_unet.pt"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss, train_iou, train_dice = (
            run_epoch(
                model=model,
                data_loader=train_loader,
                loss_function=loss_function,
                device=device,
                optimizer=optimizer,
            )
        )

        with torch.no_grad():
            val_loss, val_iou, val_dice = (
                run_epoch(
                    model=model,
                    data_loader=val_loader,
                    loss_function=loss_function,
                    device=device,
                    optimizer=None,
                )
            )

        scheduler.step(val_loss)

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        history_row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_iou": train_iou,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
            "learning_rate": current_learning_rate,
        }

        history.append(history_row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_dice": val_dice,
            "validation_iou": val_iou,
            "image_height": args.height,
            "image_width": args.width,
            "base_channels": args.base_channels,
        }

        torch.save(
            checkpoint,
            latest_model_path,
        )

        if val_dice > best_validation_dice:
            best_validation_dice = val_dice

            torch.save(
                checkpoint,
                best_model_path,
            )

            best_indicator = " [BEST]"
        else:
            best_indicator = ""

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss {train_loss:.4f} | "
            f"Train IoU {train_iou:.4f} | "
            f"Train Dice {train_dice:.4f} | "
            f"Val loss {val_loss:.4f} | "
            f"Val IoU {val_iou:.4f} | "
            f"Val Dice {val_dice:.4f} | "
            f"LR {current_learning_rate:.6f}"
            f"{best_indicator}"
        )

    history_path = (
        OUTPUT_DIRECTORY
        / "training_history.csv"
    )

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_iou",
                "train_dice",
                "val_loss",
                "val_iou",
                "val_dice",
                "learning_rate",
            ],
        )

        writer.writeheader()
        writer.writerows(history)

    save_training_plot(
        history=history,
        output_path=(
            OUTPUT_DIRECTORY
            / "loss_curve.png"
        ),
    )

    print()
    print("========================================")
    print("TRAINING COMPLETE")
    print("========================================")
    print(
        f"Best validation Dice: "
        f"{best_validation_dice:.4f}"
    )
    print(f"Best model: {best_model_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()