# TRAINING_REPORT.md

# Training Report

## Overview

A custom U-Net implemented in PyTorch was trained for binary semantic segmentation of the `main_object` class. The objective was to produce accurate object masks for downstream contour extraction and millimetre-level measurement.

---

## Model

| Property | Value |
|---|---|
| Framework | PyTorch |
| Architecture | Custom U-Net |
| Input Channels | 3 (RGB) |
| Output Channels | 1 (Binary Mask) |
| Base Channels | 32 |

The network follows the standard encoder-decoder U-Net architecture with skip connections and transposed convolutions for upsampling.

---

## Training Configuration

| Parameter | Value |
|---|---:|
| Input Resolution | 192 × 256 px |
| Epochs | 40 |
| Batch Size | 4 |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Loss Function | Dice + BCE Loss |
| LR Scheduler | ReduceLROnPlateau |
| Random Seed | 42 |

Training parameters were extracted directly from the training implementation.

---

## Data Preparation

Training images were resized to the network input resolution and normalized using ImageNet mean and standard deviation.

Data augmentation included:

- Random horizontal flip
- Random brightness adjustment
- Random contrast adjustment

Validation images were not augmented.

---

## Training Strategy

The model was trained on the training split and evaluated after every epoch on the validation split.

For each epoch, the following metrics were recorded:

- Training Loss
- Validation Loss
- Training IoU
- Validation IoU
- Training Dice
- Validation Dice

The checkpoint with the highest validation Dice score was saved as:

```text
models/outputs/best_unet.pt
```

The most recent checkpoint was also saved as:

```text
models/outputs/latest_unet.pt
```

Training history and the loss curve were exported for later analysis.

---

## Training Curve

The loss curve demonstrates a consistent reduction in training loss throughout the 40 epochs. Validation loss follows a similar trend with a few temporary spikes, after which it quickly recovers and continues decreasing. This behaviour is typical when using a small custom dataset with online augmentation.

The final epochs show convergence, indicating that the model learned a stable segmentation solution without obvious divergence.

Example output:

```text
models/outputs/loss_curve.png
```

---

## Final Evaluation Results

### Mean Metrics

| Metric | Score |
|---|---:|
| IoU | **0.9619** |
| Dice | **0.9803** |
| Precision | **0.9785** |
| Recall | **0.9825** |
| F1 Score | **0.9803** |
| Pixel Accuracy | **0.9930** |

### Global Metrics

| Metric | Score |
|---|---:|
| IoU | **0.9724** |
| Dice | **0.9860** |
| Precision | **0.9819** |
| Recall | **0.9902** |
| Pixel Accuracy | **0.9930** |

These results indicate excellent segmentation quality and accurate boundary prediction on the held-out test dataset.

---

## Generated Outputs

Training generates the following outputs:

```text
models/outputs/
├── best_unet.pt
├── latest_unet.pt
├── training_history.csv
├── loss_curve.png
└── test_predictions/
```

---

## Conclusion

The custom U-Net converged successfully within 40 epochs and achieved high segmentation accuracy across all evaluation metrics. The resulting model provides reliable binary masks that serve as the foundation for contour extraction and homography-based physical measurement.
