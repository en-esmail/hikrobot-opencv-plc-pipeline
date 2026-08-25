# Detection Improvements And Tuning Guide

This document summarizes the detection changes made to the project, explains the important threshold parameters, and gives guidance for adding new shapes or colors later.

## Current Detection Scope

The active detector is now limited to these shapes:

- `Square`
- `Circle`
- `Triangle`

The active detector accepts only these colors:

- `Red`
- `Yellow`
- `Blue`

Any other shape or weak color match is ignored.

## Project Structure After Separation

The project is now split into focused modules:

- `main.py`: live camera loop, keyboard controls, and sorting trigger.
- `camera_io.py`: Hikrobot SDK setup and frame conversion to OpenCV BGR images.
- `vision_detection.py`: OpenCV preprocessing, shape detection, color detection, and drawing labels on frames.
- `calibration.py`: pixel-to-robot coordinate conversion.
- `plc_controller.py`: PLC socket communication and robot pick/place movement.
- `sorting.py`: sorting decision and target/reject sorting flow.
- `constants.py`: shared configuration values for camera, detection, colors, calibration, PLC, and sorting.
- `vision_sorter_pipeline.py`: backward-compatible wrapper for older commands that still run this filename.

New detection changes should be made mainly in `vision_detection.py` and `constants.py`.

## Modification Steps Completed

1. Fixed the missing/unreachable `classify_shape()` issue.

   The old shape-classification logic was located after a `return` statement in the monolithic file, so calls to `classify_shape(contour)` could fail or use stale code depending on which file was executed.

   The active implementation is now in:

   ```text
   vision_detection.py
   ```

2. Restricted shape detection to only the required shapes.

   The allowed shape list is now:

   ```python
   VALID_SHAPES = ["Square", "Circle", "Triangle"]
   ```

   This is configured in:

   ```text
   constants.py -> SortingRulesConstants.VALID_SHAPES
   ```

3. Removed unsupported shape outputs.

   The detector no longer returns:

   - `Rectangle`
   - `Pentagon`
   - `Hexagon`

   A four-sided object is accepted only if its aspect ratio fits the square range. Otherwise it is rejected.

4. Improved shape validation.

   The detector now checks:

   - contour area
   - contour perimeter
   - convex-hull solidity
   - polygon side count
   - circle circularity
   - square aspect ratio

   These checks reduce false detections from shadows, rough edges, merged objects, and background noise.

5. Improved object center calculation.

   The pick point now uses contour moments:

   ```python
   moments = cv2.moments(contour)
   cx = moments["m10"] / moments["m00"]
   cy = moments["m01"] / moments["m00"]
   ```

   This is usually more accurate than using the center of the bounding box, especially for triangles.

6. Improved color confidence.

   The color detector now calculates how many pixels inside the object mask match each HSV color range.

   If the best color match is too weak, it returns:

   ```python
   "Unknown"
   ```

   Unknown colors are rejected because they are not in:

   ```python
   VALID_COLORS = ["Red", "Yellow", "Blue"]
   ```

7. Fixed frame-label matching.

   The display overlay now calculates shape and color from the same contour it draws. This avoids labels being attached to the wrong object when contour order changes.

8. Updated the split structure.

   Detection logic was applied to `vision_detection.py`, which is what `main.py` imports.

   The old `vision_sorter_pipeline.py` file was changed into a compatibility wrapper so old commands still work, while the real implementation stays in the separated modules.

9. Updated camera constants usage.

   `main.py` now uses values from `CameraConstants` instead of hard-coded values for exposure, gain, frame timeout, sleep delay, and display window size.

10. Fixed split-module PLC dependencies.

   `plc_controller.py` now imports `pixel_to_robot()` from `calibration.py` and uses `SortingRulesConstants` directly for target/reject checks.

11. Added dependency documentation.

   Added:

   ```text
   requirements.txt
   ```

   with:

   ```text
   numpy
   opencv-python
   ```

## Important Files Changed

- `vision_detection.py`
- `constants.py`
- `main.py`
- `plc_controller.py`
- `sorting.py`
- `vision_sorter_pipeline.py`
- `README.md`
- `.gitignore`
- `requirements.txt`
- `DETECTION_IMPROVEMENTS.md`

## Detection Flow

The detection pipeline works in this order:

1. Receive a BGR frame from the camera.
2. Blur the image to reduce noise.
3. Convert the frame to HSV for color classification.
4. Convert the frame to grayscale for contour detection.
5. Threshold the grayscale image.
6. Apply morphology open and close operations to clean the mask.
7. Find external contours.
8. Reject contours that are too small.
9. Classify the contour shape.
10. Reject shapes outside `Square`, `Circle`, and `Triangle`.
11. Create a filled mask for the object.
12. Classify the object color using HSV ranges.
13. Reject colors outside `Red`, `Yellow`, and `Blue`.
14. Calculate object centroid and angle.
15. Return detection data for drawing and sorting.

Each detection has this structure:

```python
{
    "shape": "Square",
    "color": "Yellow",
    "x": 123.4,
    "y": 567.8,
    "angle": 0.0,
}
```

## Threshold Parameters

All important detection values are in:

```text
constants.py
```

### `GAUSSIAN_BLUR_KERNEL`

Current value:

```python
GAUSSIAN_BLUR_KERNEL = (5, 5)
```

Purpose:

Smooths the image before thresholding.

Increase when:

- image noise creates rough object edges
- tiny texture details become false contours

Decrease when:

- object edges become too soft
- small triangles or corners are lost

Recommended values:

```python
(3, 3)
(5, 5)
(7, 7)
```

Use odd numbers only.

### `MORPH_KERNEL`

Current value:

```python
MORPH_KERNEL = (5, 5)
```

Purpose:

Cleans the binary threshold image.

The code applies:

- `MORPH_OPEN`: removes small white noise
- `MORPH_CLOSE`: fills small black holes

Increase when:

- noise remains after thresholding
- object masks have small holes

Decrease when:

- thin triangle tips disappear
- nearby objects merge together

Recommended values:

```python
(3, 3)
(5, 5)
(7, 7)
```

### `MIN_CONTOUR_AREA_PX2`

Current value:

```python
MIN_CONTOUR_AREA_PX2 = 500
```

Purpose:

Rejects contours that are too small to be real parts.

Increase when:

- dust, screws, marks, or reflections are detected as objects

Decrease when:

- valid small objects are missed

Tuning method:

Print or log contour areas for real objects and noise. Set this value below the smallest real object area and above the largest noise area.

### `CONTOUR_APPROX_EPSILON`

Current value:

```python
CONTOUR_APPROX_EPSILON = 0.04
```

Purpose:

Controls how strongly OpenCV simplifies contour edges into polygon corners.

Lower value means:

- more points are preserved
- better for detailed or small shapes
- may create too many corners from noisy edges

Higher value means:

- stronger simplification
- better for noisy contours
- may turn triangles/squares into the wrong side count

Recommended range:

```python
0.02 to 0.06
```

If triangles are missed, try:

```python
CONTOUR_APPROX_EPSILON = 0.03
```

If rough squares are detected with too many corners, try:

```python
CONTOUR_APPROX_EPSILON = 0.05
```

### `SQUARE_ASPECT_RATIO_MIN` And `SQUARE_ASPECT_RATIO_MAX`

Current values:

```python
SQUARE_ASPECT_RATIO_MIN = 0.80
SQUARE_ASPECT_RATIO_MAX = 1.20
```

Purpose:

Checks whether a four-sided object is close enough to equal width and height to be a square.

Tighter setting:

```python
0.90 to 1.10
```

Use this when rectangles are being falsely detected as squares.

Looser setting:

```python
0.75 to 1.25
```

Use this when real squares are rotated, perspective-distorted, or slightly misdetected.

### `CIRCLE_CIRCULARITY_MIN`

Current value:

```python
CIRCLE_CIRCULARITY_MIN = 0.75
```

Purpose:

Checks how close a contour is to a perfect circle.

Formula:

```python
circularity = 4 * pi * area / perimeter^2
```

A perfect circle is close to `1.0`.

Increase when:

- rounded squares or noisy blobs are detected as circles

Decrease when:

- real circles are missed because of shadows, bad thresholding, or partial occlusion

Recommended range:

```python
0.70 to 0.90
```

### `CIRCLE_VERTEX_COUNT_MIN`

Current value:

```python
CIRCLE_VERTEX_COUNT_MIN = 7
```

Purpose:

Prevents low-side polygons from being treated as circles.

Increase when:

- hexagons or rough polygons are detected as circles

Decrease when:

- small circles are simplified into too few contour points

Recommended range:

```python
6 to 10
```

### `POLYGON_SOLIDITY_MIN`

Current value:

```python
POLYGON_SOLIDITY_MIN = 0.90
```

Purpose:

Rejects broken, concave, noisy, or merged contours.

Formula:

```python
solidity = contour_area / convex_hull_area
```

A clean solid object is close to `1.0`.

Increase when:

- broken/merged/noisy contours are accepted

Decrease when:

- real parts have rough edges or small occlusions

Recommended range:

```python
0.85 to 0.95
```

### `COLOR_RANGES`

Current colors:

```python
COLOR_RANGES = {
    "Red": [
        (0, 70, 50, 10, 255, 255),
        (170, 70, 50, 180, 255, 255)
    ],
    "Yellow": [
        (15, 40, 50, 45, 255, 255)
    ],
    "Blue": [
        (85, 50, 50, 130, 255, 255)
    ],
}
```

Purpose:

Defines the HSV lower and upper range for each color.

HSV order is:

```text
Hue, Saturation, Value
```

Hue range in OpenCV is:

```text
0 to 180
```

Saturation and value ranges are:

```text
0 to 255
```

Red uses two ranges because red wraps around the hue boundary at `0` and `180`.

### `MIN_COLOR_MATCH_RATIO`

Current value:

```python
MIN_COLOR_MATCH_RATIO = 0.10
```

Purpose:

Requires at least 10 percent of the object mask to match the selected color.

Increase when:

- wrong colors are accepted
- shadows/reflections create false color labels

Decrease when:

- real colors are detected as `Unknown`
- lighting is uneven across the object

Recommended range:

```python
0.05 to 0.30
```

## How To Improve Accuracy

Use this tuning order:

1. Fix lighting first.

   Use stable lighting, avoid glare, and keep the background different from the object colors.

2. Save sample images.

   Press `s` in the live window to save snapshots. Capture good examples and failure examples.

3. Tune threshold and morphology.

   Adjust `GAUSSIAN_BLUR_KERNEL`, `MORPH_KERNEL`, and `MIN_CONTOUR_AREA_PX2` until object contours are clean.

4. Tune shape classification.

   Adjust `CONTOUR_APPROX_EPSILON`, `SQUARE_ASPECT_RATIO_MIN`, `SQUARE_ASPECT_RATIO_MAX`, `CIRCLE_CIRCULARITY_MIN`, `CIRCLE_VERTEX_COUNT_MIN`, and `POLYGON_SOLIDITY_MIN`.

5. Tune color classification.

   Use real snapshots to adjust HSV ranges in `COLOR_RANGES`.

6. Re-test with multiple positions.

   Test objects near the image center, edges, and corners. Perspective and lighting can change across the frame.

7. Recalibrate robot coordinates if pick position changes.

   If centroid calculation or camera setup changes, verify the pixel-to-robot calibration before running real picks.

## How To Add A New Shape

Example: add `Rectangle`.

1. Add the shape name to `VALID_SHAPES` in `constants.py`.

   ```python
   VALID_SHAPES = ["Square", "Circle", "Triangle", "Rectangle"]
   ```

2. Add any needed thresholds to `ImageProcessingConstants`.

   Example:

   ```python
   RECTANGLE_ASPECT_RATIO_MIN = 1.20
   RECTANGLE_ASPECT_RATIO_MAX = 6.00
   ```

3. Update `classify_shape()` in `vision_detection.py`.

   Example:

   ```python
   if sides == 4:
       x, y, w, h = cv2.boundingRect(approx)
       aspect_ratio = w / float(h)

       if ImageProcessingConstants.SQUARE_ASPECT_RATIO_MIN <= aspect_ratio <= ImageProcessingConstants.SQUARE_ASPECT_RATIO_MAX:
           return "Square"

       normalized_aspect_ratio = max(aspect_ratio, 1.0 / aspect_ratio)
       if ImageProcessingConstants.RECTANGLE_ASPECT_RATIO_MIN <= normalized_aspect_ratio <= ImageProcessingConstants.RECTANGLE_ASPECT_RATIO_MAX:
           return "Rectangle"
   ```

4. Update sorting rules if the new shape changes target/reject behavior.

   Current target rule is in `constants.py`:

   ```python
   TARGET_SHAPE = "Square"
   TARGET_COLOR = "Yellow"
   ```

5. Update documentation and test with real camera snapshots.

## How To Add A New Color

Example: add `Green`.

1. Add the HSV range to `COLOR_RANGES` in `constants.py`.

   ```python
   "Green": [
       (40, 50, 50, 85, 255, 255)
   ],
   ```

2. Add the color name to `VALID_COLORS`.

   ```python
   VALID_COLORS = ["Red", "Yellow", "Blue", "Green"]
   ```

3. Tune the HSV range using real images from the Hikrobot camera.

4. Update sorting rules if the new color should be a target object.

   Example:

   ```python
   TARGET_COLOR = "Green"
   ```

5. Re-test under production lighting.

## HSV Tuning Guidance

Use a saved snapshot and inspect HSV values from real object pixels.

General rules:

- Hue identifies the color family.
- Saturation identifies how colorful the pixel is.
- Value identifies brightness.
- Low saturation means gray/white/black areas can be confused with colors.
- Low value means shadows can be confused with dark colors.

When false positives happen:

- increase the lower saturation value
- increase the lower value value
- narrow the hue range

When false negatives happen:

- lower the lower saturation value
- lower the lower value value
- widen the hue range

## Validation Commands

Syntax check:

```powershell
.\.venv\Scripts\python.exe -m py_compile constants.py calibration.py camera_io.py exceptions.py health_checker.py logging_config.py main.py plc_controller.py retry_logic.py sorting.py vision_detection.py vision_sorter_pipeline.py
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the live pipeline:

```powershell
.\.venv\Scripts\python.exe main.py
```

Backward-compatible old command:

```powershell
.\.venv\Scripts\python.exe vision_sorter_pipeline.py
```

## Notes For Other PCs

After pulling changes on another PC:

```powershell
git pull origin master
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `classify_shape is not defined` appears, check that the PC is running the split-module version:

```powershell
Select-String -Path vision_detection.py -Pattern "def classify_shape"
```

The expected function is in:

```text
vision_detection.py
```

The old `vision_sorter_pipeline.py` should now be only a wrapper.

## Git Commit Reference

Initial detection improvements were committed and pushed:

```text
3bd2180 Improve shape detection accuracy
```

The split-module cleanup and this expanded guide are local changes until committed.
