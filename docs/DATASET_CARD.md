# DATASET_CARD.md

# Dataset Card

## Overview

This dataset was created specifically for the XIS AI / Computer Vision Technical Assessment. It contains images of a single custom object collected under different viewpoints and lighting conditions for semantic segmentation and physical measurement.

The dataset is entirely custom and was collected using an **Apple iPhone XS Max**.

---

# Object Information

| Property | Value |
|---|---|
| Object | Square box |
| Physical Size | 110 mm × 110 mm |
| Segmentation Class | `main_object` |
| Camera | Apple iPhone XS Max |
| Image Resolution | 960 × 1280 px |

---

# Dataset Composition

Two image groups were collected.

## Main Object Images

- 75 images
- Used for segmentation

## ArUco Images

- 45 images
- Used for segmentation and measurement

The ArUco images were divided into:

- 25 images for segmentation
- 20 images reserved for measurement validation

One validation image was excluded during final validation, resulting in **19 successfully evaluated measurement images**.

---

# Dataset Summary

| Category | Images |
|---|---:|
| Main object only | 75 |
| With ArUco marker | 45 |
| Total captured | 120 |
| Used for segmentation | 100 |
| Reserved for measurement validation | 20 |

---

# Train / Validation / Test Split

The segmentation dataset contains 100 images.

| Split | Main Images | ArUco Images | Total |
|---|---:|---:|---:|
| Train | 53 | 17 | 70 |
| Validation | 15 | 5 | 20 |
| Test | 7 | 3 | 10 |

---

# Image Collection

Images were captured with varying:

- viewing angles
- object positions
- distances
- rotations
- lighting conditions

This improves the robustness of the segmentation model.

---

# Annotation

Annotation software:

```text
CVAT
```

Export format:

```text
COCO
```

Segmentation class:

```text
main_object
```

Polygon annotations were converted into binary masks for U-Net training.

---

# Pre-processing

The following preprocessing steps were applied:

1. Convert HEIC images to JPEG (where required).
2. Resize images to the project resolution.
3. Undistort every image using the calibrated camera parameters.
4. Export annotations from CVAT.
5. Generate binary masks.
6. Split the dataset into train, validation and test sets.

---

# Repository Layout

```text
dataset/
├── scripts/
├── samples/
│   ├── raw/
│   ├── annotated/
│   ├── masks/
│   └── calibration/
├── metadata/
└── README.md
```

Large datasets, masks and annotations are hosted externally and linked from the project README.

---

# Intended Use

This dataset is intended for:

- binary semantic segmentation
- contour extraction
- object localisation
- physical measurement using an ArUco reference marker
- evaluation of segmentation quality

---

# Limitations

- Single object class
- One camera device
- Fixed image resolution
- Designed for planar object measurement
- Not intended for object detection or multi-class segmentation

---

# Conclusion

The final dataset provides a clean, custom image collection for training and evaluating a semantic segmentation model and validating a metric measurement pipeline. The combination of calibration, high-quality annotations and reserved validation images supports reliable millimetre-level measurements.
