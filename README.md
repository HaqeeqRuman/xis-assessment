# Calibrated Object Segmentation and Metric Measurement

An end-to-end computer vision pipeline for camera calibration, custom object segmentation, and real-world dimension measurement in millimetres.

The project was developed for the **XIS AI / Computer Vision Technical Hiring Assessment**. It demonstrates a complete workflow that starts with intrinsic camera calibration, continues through custom dataset collection and U-Net training, and ends with ArUco-assisted physical measurement of a segmented object.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Capabilities](#key-capabilities)
- [Selected Object](#selected-object)
- [Pipeline Overview](#pipeline-overview)
- [Main Results](#main-results)
- [Repository Structure](#repository-structure)
- [Large File Downloads](#large-file-downloads)
- [Quick Start](#quick-start)
- [Environment Requirements](#environment-requirements)
- [Camera Calibration](#camera-calibration)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Inference](#inference)
- [Metric Measurement](#metric-measurement)
- [Measurement Validation](#measurement-validation)
- [Generated Outputs](#generated-outputs)
- [Documentation](#documentation)
- [Design Decisions](#design-decisions)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Reproducibility](#reproducibility)
- [Assessment Compliance](#assessment-compliance)
- [License](#license)

---

## Project Overview

This system measures the physical width and height of a custom object from an image.

The pipeline performs the following operations:

1. Calibrates the camera using checkerboard images.
2. Removes radial and tangential lens distortion.
3. Segments the target object using a custom PyTorch U-Net.
4. Detects a known ArUco reference marker.
5. Builds a planar image-to-millimetre homography.
6. Transforms the segmented object contour into metric coordinates.
7. Measures the object using a minimum-area rotated rectangle.
8. Produces an annotated image, binary mask, confidence score, and JSON result.
9. Validates the measurement system against known physical dimensions.

The implementation is modular, reproducible, and suitable for an industrial-style measurement workflow.

---

## Key Capabilities

- Intrinsic camera calibration with OpenCV
- Lens distortion correction
- Custom image collection and annotation
- COCO annotation processing
- Train, validation, and test dataset splitting
- Binary mask generation
- Custom U-Net training in PyTorch
- Segmentation evaluation using IoU, Dice, precision, recall, F1, and accuracy
- Single-image inference
- ArUco marker detection
- Perspective-aware planar metric measurement
- Width and height estimation in millimetres
- Batch measurement validation
- MAE and MPE calculation
- Annotated result generation
- JSON and CSV reporting

---

## Selected Object

The selected object is a square box with known physical dimensions:

```text
Width:  110 mm
Height: 110 mm
```

The object was selected because it:

- is easily available;
- has clear geometric boundaries;
- has approximately planar visible surfaces;
- is suitable for polygon-based segmentation;
- allows direct physical verification using a ruler;
- provides a simple ground truth for measurement accuracy.

The segmentation class is:

```text
main_object
```

---

## Pipeline Overview

```text
Checkerboard Images
        |
        v
Intrinsic Camera Calibration
        |
        v
Camera Matrix + Distortion Coefficients
        |
        v
Image Undistortion
        |
        +-----------------------------+
        |                             |
        v                             v
Dataset Collection               New Input Image
        |                             |
        v                             v
CVAT Annotation                Image Undistortion
        |                             |
        v                             v
COCO Export                  U-Net Segmentation
        |                             |
        v                             v
Dataset Split                Binary Object Mask
        |                             |
        v                             v
Mask Generation              ArUco Marker Detection
        |                             |
        v                             v
U-Net Training               Planar Homography
        |                             |
        v                             v
Model Evaluation             Metric Contour Transform
                                      |
                                      v
                              Width and Height in mm
                                      |
                                      v
                         Annotated Image + JSON Result
```

---

## Main Results

### Camera Calibration

| Metric | Result |
|---|---:|
| Checkerboard squares | 10 x 8 |
| Inner corners | 9 x 7 |
| Square size | 20 mm |
| RMS calibration error | 0.4905 px |
| Mean reprojection RMSE | 0.4818 px |

Camera intrinsic matrix:

```text
[[991.55147509,   0.00000000, 490.89708498],
 [  0.00000000, 991.22887419, 634.45582119],
 [  0.00000000,   0.00000000,   1.00000000]]
```

Distortion coefficients:

```text
[ 0.285949717,
 -1.434332420,
  0.00118800109,
 -0.000769498297,
  2.100589750 ]
```

### Segmentation Performance

Final test-set results:

| Metric | Mean Result |
|---|---:|
| IoU | 0.9619 |
| Dice | 0.9803 |
| Precision | 0.9785 |
| Recall | 0.9825 |
| F1 | 0.9803 |
| Accuracy | 0.9930 |

Global pixel-level results:

| Metric | Global Result |
|---|---:|
| IoU | 0.9724 |
| Dice | 0.9860 |
| Precision | 0.9819 |
| Recall | 0.9902 |
| Accuracy | 0.9930 |

### Measurement Validation

The measurement system was evaluated on 19 reserved images containing the reference marker.

| Metric | Result |
|---|---:|
| Total validation images | 19 |
| Successful measurements | 19 |
| Failed measurements | 0 |
| Success rate | 100.00% |
| Long-side MAE | 5.5836 mm |
| Short-side MAE | 3.2825 mm |
| Overall MAE | 4.4330 mm |
| Long-side MPE | 5.0760% |
| Short-side MPE | 2.9840% |
| Overall MPE | 4.0300% |
| Mean segmentation confidence | 98.86% |

Example result:

```text
Ground truth: 110.00 x 110.00 mm
Prediction:   110.62 x 110.23 mm
```

---

## Repository Structure

The final repository follows the assessment structure while keeping scripts, reports, and outputs grouped by pipeline stage.

```text
project-root/
|
|-- calibration/
|   |-- calibrate_camera.py
|   |-- undistort_images.py
|   |-- outputs/
|   |   |-- camera_calibration.npz
|   |   |-- calibration_summary.json
|   |   `-- sample_undistorted/
|   `-- README.md
|
|-- dataset/
|   |-- scripts/
|   |   |-- convert_heic.py
|   |   |-- resize_images.py
|   |   |-- split_coco_dataset.py
|   |   `-- create_masks.py
|   |-- samples/
|   |   |-- calibration/
|   |   |-- raw/
|   |   |-- annotated/
|   |   `-- masks/
|   |-- metadata/
|   |   |-- dataset_split.json
|   |   `-- class_distribution.json
|   `-- README.md
|
|-- models/
|   |-- train_unet.py
|   |-- evaluate_model.py
|   |-- predict_test_set.py
|   |-- configs/
|   |   `-- unet_config.json
|   |-- outputs/
|   |   |-- metrics.json
|   |   |-- training_history.json
|   |   |-- plots/
|   |   `-- test_predictions/
|   `-- README.md
|
|-- inference/
|   |-- run_inference.py
|   |-- samples/
|   `-- outputs/
|
|-- measurement/
|   |-- measure_object.py
|   |-- validate_measurements.py
|   |-- outputs/
|   |   |-- measurement_validation.csv
|   |   |-- measurement_validation_summary.json
|   |   `-- sample_results/
|   `-- README.md
|
|-- docs/
|   |-- CALIBRATION_REPORT.md
|   |-- DATASET_CARD.md
|   |-- TRAINING_REPORT.md
|   |-- MEASUREMENT_REPORT.md
|   |-- PIPELINE_ARCHITECTURE.md
|   `-- MODULE_REFERENCE.md
|
|-- .gitignore
|-- requirements.txt
|-- SETUP.md
`-- README.md
```

### Large files excluded from GitHub

The following should not be committed to the repository:

```text
calibration image collection
raw dataset images
full undistorted dataset
full COCO annotation export
generated training masks
trained model weights
large prediction collections
```

Only small representative samples should remain in the repository.

The complete large files must be hosted on Google Drive or OneDrive and linked below.

---

## Large File Downloads



| Resource | Description | Download |
|---|---|---|
| Dataset Images | Full checkerboard image collection used for intrinsic calibration | [https://drive.google.com/drive/folders/1HLIQDQJzazsaAJvh2DpSUpbicofkjoBu?usp=drive_link]() |
|

---

## Quick Start

### 1. Clone the repository

```bash
git clone REPLACE_WITH_REPOSITORY_URL
cd REPLACE_WITH_REPOSITORY_NAME
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download required large files

Download the trained model and dataset resources from the links in [Large File Downloads](#large-file-downloads).

Place the best model checkpoint at:

```text
models/outputs/best_unet.pt
```

Place the calibration file at:

```text
calibration/outputs/camera_calibration.npz
```

### 5. Run single-image inference

```bash
python inference/run_inference.py --image path/to/input.jpg
```

For an image that is already undistorted:

```bash
python inference/run_inference.py \
  --image path/to/undistorted_input.jpg \
  --already-undistorted
```

### 6. Measure an object

```bash
python measurement/measure_object.py \
  --image path/to/input_with_aruco.jpg
```

For an already-undistorted image:

```bash
python measurement/measure_object.py \
  --image path/to/undistorted_input_with_aruco.jpg \
  --already-undistorted
```

### 7. Validate all measurement images

```bash
python measurement/validate_measurements.py \
  --already-undistorted
```

For detailed setup and execution instructions, see [SETUP.md](SETUP.md).

---

## Environment Requirements

Recommended environment:

```text
Python: 3.10 or 3.11
Operating system: Windows, Linux, or macOS
GPU: Optional
CPU inference: Supported
```

Main libraries:

- OpenCV
- opencv-contrib-python
- PyTorch
- torchvision
- NumPy
- Pillow
- matplotlib

The ArUco module requires:

```bash
pip install opencv-contrib-python
```

Do not install both `opencv-python` and `opencv-contrib-python` in the same environment unless their versions are compatible.

---

## Camera Calibration

The camera was calibrated using a printed checkerboard.

Configuration:

```text
Checkerboard squares: 10 x 8
Inner corners:        9 x 7
Square size:          20 mm
Calibration images:   20 or more
```

The calibration stage estimates:

- focal lengths;
- optical centre;
- radial distortion coefficients;
- tangential distortion coefficients;
- reprojection error.

The stored calibration file is:

```text
calibration/outputs/camera_calibration.npz
```

All images used for segmentation measurement must be undistorted before metric conversion.

Detailed calibration theory, parameters, and results are documented in:

[docs/CALIBRATION_REPORT.md](docs/CALIBRATION_REPORT.md)

---

## Dataset Preparation

The original collected dataset contained:

```text
75 images without ArUco marker
45 images with ArUco marker
```

The ArUco images were separated into:

```text
25 images for segmentation dataset use
20 images reserved for measurement validation
```

One validation image was not included in the final batch, resulting in 19 successfully processed validation images.

The segmentation dataset was split approximately as follows:

| Split | Main-only Images | ArUco Images | Total |
|---|---:|---:|---:|
| Train | 53 | 17 | 70 |
| Validation | 15 | 5 | 20 |
| Test | 7 | 3 | 10 |
| Total | 75 | 25 | 100 |

Annotations were created in CVAT and exported in COCO format.

Annotation class:

```text
main_object
```

Binary masks were generated from polygon annotations for U-Net training.

Detailed dataset information is available in:

[docs/DATASET_CARD.md](docs/DATASET_CARD.md)

---

## Model Training

A custom U-Net was implemented in PyTorch.

The model was selected because U-Net:

- is designed for dense pixel-level segmentation;
- performs well on relatively small custom datasets;
- preserves spatial details through skip connections;
- is lightweight enough for CPU inference;
- is not a Roboflow or Ultralytics YOLO model;
- provides direct binary mask output suitable for contour-based measurement.

Training is run with:

```bash
python models/train_unet.py
```

The exact hyperparameters, augmentations, loss configuration, model architecture, and training curves are documented in:

[docs/TRAINING_REPORT.md](docs/TRAINING_REPORT.md)

---

## Model Evaluation

Evaluate the trained model with:

```bash
python models/evaluate_model.py
```

Generate prediction visualisations for the held-out test set:

```bash
python models/predict_test_set.py
```

Expected prediction outputs include:

- original image;
- ground-truth mask;
- predicted mask;
- colour overlay;
- side-by-side comparison.

The project reports semantic segmentation metrics rather than object-detection mAP as its primary evaluation because the model produces a binary segmentation mask rather than detection boxes or instance confidence-ranked predictions.

Reported metrics include:

- IoU;
- Dice coefficient;
- precision;
- recall;
- F1;
- pixel accuracy.

---

## Inference

The inference pipeline accepts one image and performs:

```text
input image
    |
    v
optional camera undistortion
    |
    v
image preprocessing
    |
    v
U-Net inference
    |
    v
probability map
    |
    v
binary mask
    |
    v
largest connected component
    |
    v
mask overlay and output files
```

Run:

```bash
python inference/run_inference.py --image path/to/image.jpg
```

Typical outputs:

```text
inference/outputs/
|-- image_processed.jpg
|-- image_probability.png
|-- image_mask.png
|-- image_overlay.jpg
`-- image_comparison.jpg
```

---

## Metric Measurement

The measurement pipeline requires a visible ArUco marker with known physical size.

Reference configuration:

```text
Dictionary: DICT_4X4_50
Marker ID:  8
Marker size: 39 mm
```

### Measurement method

The final method uses the four detected marker corners to construct a planar projective transformation.

The marker corners in image coordinates are mapped to:

```text
(0, 0)
(39, 0)
(39, 39)
(0, 39)
```

These coordinates are expressed in millimetres.

OpenCV computes the homography:

```text
metric_point ~ H x image_point
```

The complete object contour is then transformed from image coordinates into the marker-defined metric plane.

A minimum-area rotated rectangle is fitted to the transformed contour:

```python
cv2.minAreaRect(metric_contour)
```

Its side lengths are returned directly in millimetres.

This approach is more robust to perspective than using one average pixels-per-millimetre ratio.

Run:

```bash
python measurement/measure_object.py \
  --image path/to/image_with_marker.jpg
```

Outputs include:

- processed image;
- binary mask;
- annotated measurement image;
- JSON measurement result;
- long side in millimetres;
- short side in millimetres;
- mean segmentation confidence.

Detailed derivation and accuracy analysis are available in:

[docs/MEASUREMENT_REPORT.md](docs/MEASUREMENT_REPORT.md)

---

## Measurement Validation

Batch validation is run with:

```bash
python measurement/validate_measurements.py \
  --already-undistorted
```

Default validation directory:

```text
dataset/undistorted/measurement_validation
```

Generated files:

```text
measurement/outputs/measurement_validation.csv
measurement/outputs/measurement_validation_summary.json
```

The CSV records:

- image name;
- success or failure status;
- ground-truth dimensions;
- predicted dimensions;
- absolute errors;
- percentage errors;
- confidence;
- failure reason.

### Error definitions

For one measured dimension:

```text
Absolute Error = |Predicted - Ground Truth|
```

```text
Percentage Error = Absolute Error / Ground Truth x 100
```

Mean absolute error:

```text
MAE = mean of all absolute errors
```

Mean percentage error:

```text
MPE = mean of all percentage errors
```

In this project, MPE is reported as the mean absolute percentage error rather than a signed bias measure.

---

## Generated Outputs

The repository may contain a limited number of small representative outputs.

Full outputs should be stored externally.

Typical generated artifacts include:

```text
calibration/outputs/
models/outputs/
inference/outputs/
measurement/outputs/
```

Recommended GitHub samples:

- one checkerboard detection image;
- one undistorted calibration example;
- three segmentation comparisons;
- two single-image inference outputs;
- two annotated measurement examples;
- measurement validation CSV;
- summary JSON;
- reduced-resolution plots.

---

## Documentation

| Document | Purpose |
|---|---|
| [README.md](README.md) | Project overview, results, quick start, and repository guide |
| [SETUP.md](SETUP.md) | Complete environment setup and execution instructions |
| [CALIBRATION_REPORT.md](docs/CALIBRATION_REPORT.md) | Camera-calibration method, intrinsic parameters, and reprojection error |
| [DATASET_CARD.md](docs/DATASET_CARD.md) | Object, collection strategy, annotations, splits, and limitations |
| [TRAINING_REPORT.md](docs/TRAINING_REPORT.md) | U-Net architecture, hyperparameters, training process, and metrics |
| [MEASUREMENT_REPORT.md](docs/MEASUREMENT_REPORT.md) | Homography-based measurement method, validation table, MAE, and MPE |
| [PIPELINE_ARCHITECTURE.md](docs/PIPELINE_ARCHITECTURE.md) | End-to-end system architecture and data flow |
| [MODULE_REFERENCE.md](docs/MODULE_REFERENCE.md) | Script purposes, important functions, inputs, and outputs |

---

## Design Decisions

### Custom U-Net instead of YOLO

A custom U-Net was selected because this task requires accurate foreground masks for contour-based measurement.

The model directly predicts one probability value for every pixel. This is more appropriate than an object detector when the downstream task relies on the exact visible object boundary.

### Binary segmentation

Only one foreground class is required:

```text
main_object
```

The system therefore uses binary segmentation rather than multiclass segmentation.

### Largest connected component

Small isolated prediction regions are removed by keeping the largest connected foreground component. This reduces the influence of noise during contour measurement.

### ArUco reference marker

An ArUco marker provides:

- automatic detection;
- known corner order;
- known physical size;
- stable planar correspondences;
- an image-to-metric reference.

### Homography instead of one scale ratio

A single pixels-per-millimetre value assumes uniform scale throughout the image.

Perspective causes scale to vary by image location. The final implementation therefore transforms the object contour into a metric plane using the marker corners before measuring it.

### Undistortion before measurement

Lens distortion changes image geometry, especially near image boundaries. Measurement images are therefore undistorted using the stored intrinsic calibration before reference detection and contour conversion.

---

## Assumptions and Limitations

The measurement result is valid under the following assumptions:

1. The ArUco marker is fully visible.
2. The marker is detected as ID 8.
3. The printed marker side length is exactly 39 mm.
4. The marker is flat and not bent.
5. The marker and the measured object surface are on the same physical plane.
6. The visible object boundary represents the dimensions being measured.
7. The input image is captured by the calibrated camera.
8. The correct calibration parameters are used.
9. The object is sufficiently visible and not heavily occluded.
10. The segmentation model recognises the object correctly.

Important limitations:

- A marker placed on a different depth plane can introduce scale error.
- Homography corrects planar perspective only.
- Three-dimensional height differences are not modelled.
- Marker blur, glare, cropping, or obstruction can prevent detection.
- Shadows and background similarity can affect mask boundaries.
- `minAreaRect` measures the visible projected contour, not hidden geometry.
- Calibration parameters are specific to the camera and capture configuration.
- Digital zoom, focus changes, or image resizing before calibration handling may reduce accuracy.
- A square ground-truth object makes side assignment simple; non-square objects require consistent orientation handling.

---

## Reproducibility

For reproducible results:

- use the same camera calibration file;
- use the same trained checkpoint;
- use the same image preprocessing dimensions;
- keep the segmentation threshold unchanged;
- use the same ArUco dictionary and marker size;
- do not resize images differently after undistortion;
- run commands from the repository root;
- preserve train, validation, and test split metadata;
- record package versions in `requirements.txt`;
- set random seeds in training.

The project should be tested from a clean virtual environment before submission.

---

## Assessment Compliance

This repository is structured to satisfy the required assessment stages.

### Step 1: Camera Calibration and Dataset

- 20 or more calibration images
- checkerboard-based intrinsic calibration
- documented camera matrix
- documented distortion coefficients
- reprojection error below 0.5 px
- 70 or more custom object images
- CVAT annotation
- train, validation, and test splits
- undistorted dataset preparation

### Step 2: Segmentation Model

- custom non-YOLO architecture
- PyTorch U-Net implementation
- reproducible training script
- held-out test evaluation
- IoU, Dice, precision, recall, F1, and accuracy
- test prediction visualisations
- standalone inference script

### Step 3: Pixel-to-Millimetre Measurement

- known physical reference object
- ArUco reference detection
- calibrated and undistorted measurement input
- perspective-aware metric conversion
- output width and height in millimetres
- confidence output
- 19-image physical validation
- MAE and MPE reporting
- single-image end-to-end demonstration

### Repository and submission discipline

- lightweight Git repository
- large assets hosted on Google Drive or OneDrive
- professional documentation
- modular scripts
- meaningful output files
- incremental Git history

---

## License

This repository was created for a technical hiring assessment.

Unless a separate licence is added, the code and documentation should be treated as assessment material and not as a general-purpose open-source release.
