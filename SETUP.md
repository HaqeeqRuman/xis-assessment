# XIS Assessment Setup Guide

This guide explains how to set up the project and run the pretrained inference and measurement pipeline without retraining the U-Net model.

## 1. Requirements

Install:

- Python 3.12
- Git
- Windows PowerShell

The repository should include these pretrained artifacts:

```text
models/outputs/best_unet.pt
calibration/outputs/camera_calibration.npz
calibration/outputs/calibration_report.json
```

The repository should also include the sample inference image:

```text
dataset/undistorted/measurement_validation/aruco_0002.jpg
```

Training is not required before inference.

## 2. Open the project

```powershell
Set-Location "E:\XIS-ASSESSMENT"
```

## 3. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Verify required files

```powershell
$requiredFiles = @(
    "models\outputs\best_unet.pt",
    "calibration\outputs\camera_calibration.npz",
    "calibration\outputs\calibration_report.json",
    "dataset\undistorted\measurement_validation\aruco_0002.jpg"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "[OK] $file"
    }
    else {
        Write-Host "[MISSING] $file"
    }
}
```

All required files should display `[OK]`.

## 6. Run inference and measurement

Use the included sample image:

```powershell
python inference\run_inference.py --image "dataset\undistorted\measurement_validation\aruco_0002.jpg"
```

The pipeline should:

1. Load the pretrained U-Net model.
2. Load the saved camera calibration.
3. Read the undistorted input image.
4. Predict the object segmentation mask.
5. Detect the ArUco reference marker.
6. Calculate the object measurements.
7. Save the generated images and measurement results.

Expected output folders:

```text
inference/outputs/
measurement/outputs/
```

## 7. Run inference on another image

```powershell
python inference\run_inference.py --image "path\to\your\image.jpg"
```

Use quotation marks when the path contains spaces.

## 8. Optional individual stages

### Camera calibration

```powershell
python calibration\calibrate_camera.py
```

### Batch undistortion

```powershell
python calibration\batch_undistort.py
```

### Model training

Training is optional because the pretrained model is included.

```powershell
python models\train_unet.py
```

### Test-set evaluation

```powershell
python models\evaluate_test_set.py
```

### Test-set prediction

```powershell
python models\predict_test_set.py
```

### Measurement validation

```powershell
python measurement\validate_measurements.py
```

### ArUco detection test

```powershell
python measurement\test_aruco_detection.py
```

## 9. Deactivate the environment

```powershell
deactivate
```

## Troubleshooting

### Python is not recognized

```powershell
python --version
```

You can also use the Python launcher:

```powershell
py -3.12 -m venv .venv
```

### A module is missing

```powershell
python -m pip install -r requirements.txt
```

### The pretrained model is missing

Confirm that this file exists:

```text
models/outputs/best_unet.pt
```

### Camera calibration is missing

Confirm that this file exists:

```text
calibration/outputs/camera_calibration.npz
```

### The sample image is missing

Confirm that this file exists:

```text
dataset/undistorted/measurement_validation/aruco_0002.jpg
```

### CUDA is unavailable

The pipeline can run on CPU if the code selects the available PyTorch device, although CPU inference will be slower.
