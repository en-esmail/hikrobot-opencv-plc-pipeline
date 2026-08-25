# Vision-Guided PLC Sorter

This repository contains a unified Python pipeline that completely replaces external software (like Vision Master) by combining Hikrobot camera control, OpenCV image processing, and PLC-driven robotic sorting into a single system.

## Features

* **Direct Camera Integration:** Uses the Hikrobot MVS SDK to capture raw frames and converts them directly into OpenCV BGR8 format using C-memory management (`ctypes`).
* **Shape & Color Detection:** Identifies red, yellow, and blue objects using HSV color ranges, then classifies only squares, circles, and triangles with contour geometry.
* **Socket-Based PLC Control:** Communicates directly with the PLC via standard TCP sockets over the local network (IP: 192.168.6.10, Port: 2023).
* **Automated Pick-and-Place:** Executes hardware movements using specific string protocols (e.g., `plc,6` for readiness handshakes, and `plc,4` / `plc,5` for suction cup control).

---

## Prerequisites & Installation

To run this pipeline, you must configure both the Hikrobot drivers and your Python environment.

* The Hikrobot MVS SDK must be downloaded and installed from the official Hikrobot website.
* The `MvImport` wrapper folder (found in the SDK's `Development/Samples/Python/MvImport` directory) must be copied and placed directly next to the scripts in this repository.
* Install the required Python libraries with `pip install -r requirements.txt`.
* Ensure your PC's network adapter is on the same subnet as the PLC (e.g., `192.168.6.X`, where X is any number except 10).

---

## Project Structure

* `main.py`: live camera loop and keyboard controls.
* `camera_io.py`: Hikrobot SDK setup and frame conversion.
* `vision_detection.py`: OpenCV preprocessing, shape detection, color detection, and frame annotation.
* `calibration.py`: pixel-to-robot coordinate transform.
* `plc_controller.py`: PLC socket communication and robot pick/place motions.
* `sorting.py`: sorting decisions and target/reject orchestration.
* `constants.py`: camera, image-processing, color, PLC, and sorting parameters.
* `vision_sorter_pipeline.py`: backward-compatible wrapper for older commands.

---

## Hardware Calibration

Because the camera operates in pixel coordinates and the robot operates in physical millimeters, you must calibrate the system before running a live sorting sequence.

* Collect at least three known correspondences between image pixels and robot X/Y coordinates.
* Fit an affine transform with `cv2.getAffineTransform` for three points or `cv2.estimateAffine2D` for four or more points.
* Update `CalibrationConstants.PIXEL_TO_ROBOT_MATRIX` in `constants.py`.
* **SAFETY WARNING:** This script moves the real physical robot. Clear the workspace of obstructions and always keep a hand on the physical Emergency Stop (E-Stop) button.

---

## Running the Sorting Pipeline

Once calibration is complete and your matrix is hard-coded into the pipeline, you can begin sorting.

* Run `main.py` to initiate the camera connection and open the live OpenCV video feed.
* The script runs in a continuous free-run mode, drawing bounding boxes and labels on detected objects in real-time.
* To trigger the sorting routine, press the **'t'** key on your keyboard.
* Triggering the routine will freeze the live video, send the target coordinates to the PLC, execute the physical pick-and-place sequence for every detected object, and resume the video once finished.

**Keyboard Controls:**

* **'t'** : Trigger the PLC sorting sequence.
* **'s'** : Save a snapshot of the current video frame to your local drive.
* **'q'** : Safely break the loop, release the camera hardware, close the device connection, and exit the program.
