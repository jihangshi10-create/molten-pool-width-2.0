# Robust Melt Pool Characterization for Irregular Motion

This repository contains a C++/OpenCV implementation for robust melt pool characterization in Laser Metal Deposition (LMD), with a focus on **irregular melt pool motion**. The algorithm is designed to estimate:

- **melt pool width**
- **spatial cooling rate**

from grayscale coaxial camera images.

Unlike simple fixed-threshold methods, this implementation is intended to remain robust under:

- irregular or changing motion direction
- varying illumination
- different melt pool brightness levels
- image noise and thermal background interference

The implementation is based on the irregular-motion algorithm described in my bachelor thesis and adapted into a practical C++ workflow using OpenCV. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## Features

- Perspective correction using camera intrinsic/extrinsic parameters
- Optional image downsampling for faster processing
- Robust grayscale preprocessing:
  - bilateral filtering
  - CLAHE
  - adaptive three-segment LUT
- Automatic ROI extraction using Otsu thresholding
- Melt pool width estimation from the **short side of the minimum-area rotated rectangle**
- Cooling rate estimation based on the **movement direction inferred from the rotated rectangle**
- Visualization and per-frame result export

These steps correspond closely to the irregular-motion pipeline in the thesis: contrast enhancement with bilateral filtering, CLAHE, and adaptive LUT; Otsu-based ROI extraction; largest contour selection; rotated minimum bounding rectangle fitting; and cooling-rate estimation from the midpoint of the short side along the long-axis direction. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

---

## Background

In Laser Metal Deposition, the geometry and thermal behavior of the melt pool strongly affect process stability and final part quality. In particular:

- **melt pool width** influences track geometry and consistency
- **cooling rate** affects solidification behavior and microstructure

The thesis develops two algorithms for melt pool analysis: one for linear movement and one for irregular movement. This repository focuses on the **irregular movement case**, where a simple fixed scanning direction can no longer be assumed. :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

---

## Algorithm Overview

### 1. Perspective correction
The input image can optionally be rectified using a plane-induced homography computed from the camera intrinsic matrix, rotation vector, translation vector, and plane normal. This step compensates for geometric distortion caused by camera tilt. The thesis describes this correction using calibrated camera parameters and OpenCV `warpPerspective`. :contentReference[oaicite:8]{index=8}

### 2. Preprocessing
To improve robustness under changing brightness conditions, the grayscale image is enhanced in several stages:

1. **Bilateral filtering** for denoising while preserving edges  
2. **CLAHE** for local contrast enhancement  
3. **Adaptive three-segment LUT** for highlighting the high-intensity melt pool structure while suppressing less relevant regions  

This sequence is explicitly used in the irregular-motion algorithm in the thesis, and the uploaded code implements the same pattern with bilateral filtering, CLAHE, a percentile-based LUT, and Otsu thresholding. :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

### 3. ROI extraction
After contrast enhancement, **Otsu thresholding** is applied to obtain a binary image. The largest contour is selected as the melt-pool-related region of interest. The thesis explains that, for irregular motion, the ROI should ideally include both the bright melt pool region and its thermal tail so that the motion direction can be inferred more reliably. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

### 4. Melt pool width estimation
A **minimum-area rotated rectangle** is fitted to the largest contour. The **shorter side** of this rectangle is interpreted as the melt pool width. The code then converts pixels to millimeters using a calibration factor. In the thesis, the pixel size after calibration is reported to be approximately **0.0077 mm per pixel**; the current code uses a closely related hard-coded conversion factor in the width and cooling-rate calculations. :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}

### 5. Cooling rate estimation
For irregular motion, the movement direction is inferred from the **long side direction** of the rotated rectangle. The midpoint of the short side is used as a candidate starting point, and the algorithm scans along the long-axis direction until it intersects a high-intensity region in a binary map. The longer valid segment is selected to avoid choosing the wrong motion direction. The grayscale values of the selected start and end points are then used together with scan speed and spatial distance to estimate the cooling rate. This matches the thesis description of the irregular-motion cooling-rate strategy and is reflected in the code logic that tests both short-edge midpoints and keeps the longer segment. :contentReference[oaicite:15]{index=15} :contentReference[oaicite:16]{index=16}

---

## Input

The program processes grayscale `.tif` / `.tiff` images from a folder.

At the moment, the image path is hard-coded in the source file, so you must change it manually before running:

```cpp
cv::glob("D:/Project2/1/*.tiff", fileNames, false);
cv::glob("D:/Project2/1/*.tif", tifNames, false);
## Output

For each processed frame, the program can generate:

- corrected grayscale image
- binary image
- contour/rectangle visualization image
- text log containing:
  - frame id
  - melt pool width in pixels
  - melt pool width in millimeters
  - cooling rate
  - processing time

The code saves these files into a timestamped subfolder under `output/`.

## Requirements

- C++
- OpenCV

Typical OpenCV modules used in this project include:

- `opencv_core`
- `opencv_imgproc`
- `opencv_highgui`
- `opencv_imgcodecs`

Depending on the local setup, additional OpenCV configuration may be required.

## Main Parameters

Several important options are currently defined directly in the source code:

- `SAVE_DATA`
- `SHOW_VISUALIZATION`
- `ENABLE_PERSPECTIVE_CORRECTION`
- `USE_ENHANCED_PREPROCESS`
- `ENABLE_COOLING_RATE`
- `scaleFactor`
- `scanSpeed_mmps`

Camera intrinsic/extrinsic parameters and pixel-to-millimeter conversion are also currently hard-coded in the source file.

## Notes

- This repository currently contains the core source code only.
- The runtime environment and build system are not yet fully packaged.
- Some constants, such as image path, threshold values, calibration parameters, and scale conversion factors, are still hard-coded for the current experimental setup.
- The code is intended as a research implementation and may require adaptation for other cameras, optical setups, or LMD systems.

For example, the current implementation uses Otsu thresholding for ROI extraction after enhancement, but uses a fixed high threshold again during cooling-rate estimation to approximate the bright melt pool region. The thesis describes a threshold of 220 in this stage, while the current code uses 230, so this choice may be documented or unified in future revisions.

## Future Improvements

Possible future improvements include:

- replacing hard-coded paths with command-line arguments
- exposing calibration and processing parameters via config files
- adding a CMake build system
- improving portability across Windows/Linux
- refining pixel-to-mm calibration handling
- adding batch statistics and plotting
- optimizing runtime further for real-time deployment

The thesis also discusses additional optimization ideas such as masking irrelevant image regions and alternative strategies for width extraction.

## Reference

This implementation is related to the following thesis work:

**Development of a robust algorithm for the characterization of the melt pool in laser metal deposition**  
Bachelor thesis, Ruhr University Bochum, 2025
