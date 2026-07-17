# CALIBRATION_REPORT.md

# Camera Calibration Report

## Overview

This project uses intrinsic camera calibration to remove lens distortion before segmentation and physical measurement. Calibration was performed using OpenCV with a printed checkerboard.

**Camera Used:** Apple iPhone XS Max

## Calibration Configuration

| Parameter | Value |
|---|---|
| Checkerboard squares | 10 × 8 |
| Inner corners | 9 × 7 |
| Square size | 20 mm |
| Calibration images captured | 24 |
| Successful detections | 24 |
| Failed detections | 0 |
| Image resolution | 960 × 1280 px |

All checkerboard images were successfully detected and used during calibration.

---

## Calibration Outputs

The calibration process generated:

```
calibration/outputs/
├── camera_calibration.npz
├── calibration_report.json
├── undistorted_001.jpeg
├── undistorted_001_comparison.jpeg
├── undistorted_019.jpeg
└── undistorted_019_comparison.jpeg
```

The calibration file (`camera_calibration.npz`) stores the intrinsic camera matrix and distortion coefficients used throughout the project.

---

## Camera Intrinsic Matrix

```text
[[991.55147509,   0.00000000, 490.89708498],
 [  0.00000000, 991.22887419, 634.45582119],
 [  0.00000000,   0.00000000,   1.00000000]]
```

---

## Distortion Coefficients

```text
k1 =  0.2859497171
k2 = -1.4343324183
p1 =  0.0011880011
p2 = -0.0007694983
k3 =  2.1005897526
```

These coefficients model radial and tangential lens distortion and are applied to undistort every image before segmentation or measurement.

---

## Calibration Accuracy

| Metric | Result |
|---|---:|
| OpenCV RMS calibration error | **0.4905 px** |
| Mean reprojection error | **0.4818 px** |
| Median reprojection error | **0.4821 px** |
| Minimum reprojection error | **0.2536 px** |
| Maximum reprojection error | **0.6251 px** |
| Worst image | calibration_004.jpeg |

The RMS error is below 0.5 pixels, indicating a high-quality intrinsic calibration suitable for metric measurement.

---

## Image Undistortion

All images used for segmentation, inference and measurement are undistorted using the stored calibration parameters before further processing.

Example comparison images are available:

```
calibration/outputs/undistorted_001_comparison.jpeg
calibration/outputs/undistorted_019_comparison.jpeg
```

---

## Usage in the Project

Camera calibration is performed once and reused throughout the pipeline.

The calibration parameters are used to:

- Remove lens distortion.
- Improve segmentation accuracy.
- Ensure ArUco marker detection is geometrically correct.
- Improve real-world measurement accuracy.

---

## Conclusion

The calibration process completed successfully using **24 calibration images**, with **100% checkerboard detection success** and an **RMS error of 0.4905 pixels**. The resulting camera parameters are used throughout the project to ensure reliable image geometry and accurate millimetre-level object measurements.
