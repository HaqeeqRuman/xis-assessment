# MEASUREMENT_REPORT.md

# Measurement Report

## Overview

The final stage of the project estimates the real-world dimensions of the segmented object in millimetres. The measurement pipeline combines semantic segmentation, camera calibration and ArUco marker detection to produce perspective-aware measurements.

---

# Measurement Object

| Property | Value |
|---|---|
| Object | Square box |
| Ground Truth Size | 110 mm × 110 mm |
| Reference Marker | ArUco |
| Dictionary | DICT_4X4_50 |
| Marker ID | 8 |
| Marker Size | 39 mm |

---

# Measurement Pipeline

The measurement process consists of the following steps:

1. Load the camera calibration parameters.
2. Undistort the input image.
3. Generate the object mask using the trained U-Net.
4. Detect the ArUco reference marker.
5. Compute a planar homography from image coordinates to millimetres.
6. Transform the segmented contour into metric coordinates.
7. Fit a minimum-area rectangle to the transformed contour.
8. Report the object width and height in millimetres.
9. Save annotated images and JSON results.

---

# Why Homography?

Instead of using a single pixels-per-millimetre scale, the final implementation computes a homography using the four detected ArUco corners.

This approach compensates for perspective distortion and produces more reliable measurements when the camera is not perfectly perpendicular to the object.

---

# Validation Dataset

A reserved set of ArUco images was used exclusively for measurement validation.

| Metric | Value |
|---|---:|
| Reserved validation images | 20 |
| Successfully evaluated | 19 |
| Failed | 0 |
| Success Rate | 100% |

---

# Validation Results

| Metric | Result |
|---|---:|
| Long-side MAE | **5.5836 mm** |
| Short-side MAE | **3.2825 mm** |
| Overall MAE | **4.4330 mm** |
| Long-side MPE | **5.0760 %** |
| Short-side MPE | **2.9840 %** |
| Overall MPE | **4.0300 %** |
| Mean Confidence | **98.86 %** |

Example prediction:

```text
Ground Truth : 110.00 × 110.00 mm
Prediction   : 110.62 × 110.23 mm
```

---

# Output Files

The measurement stage generates:

```text
measurement/
├── measure_object.py
├── validate_measurements.py
└── outputs/
    ├── measurement_validation.csv
    ├── measurement_validation_summary.json
    └── sample_results/
```

The validation CSV stores the predicted dimensions, confidence scores and error metrics for each processed image.

---

# Assumptions

The reported measurements assume:

- the ArUco marker is fully visible;
- the marker size is exactly 39 mm;
- the object and marker lie on the same plane;
- the image has been captured using the calibrated camera;
- the segmentation mask accurately represents the object boundary.

---

# Limitations

- Requires a visible ArUco marker.
- Designed for planar objects.
- Performance depends on segmentation quality.
- Heavy blur or occlusion may reduce accuracy.
- Calibration parameters are camera specific.

---

# Conclusion

The final measurement pipeline successfully combines camera calibration, semantic segmentation and ArUco-based planar homography to estimate real-world object dimensions. Validation on 19 reserved images achieved a **100% measurement success rate** with an **overall mean absolute error of 4.4330 mm**, demonstrating that the system is suitable for reliable millimetre-level object measurement.
