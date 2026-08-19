# Vision-Guided PLC Sorter

This repository contains a unified Python pipeline that completely replaces external software (like Vision Master) by combining Hikrobot camera control, OpenCV image processing, and PLC-driven robotic sorting into a single system.

## Features

* **Direct Camera Integration:** Uses the Hikrobot MVS SDK to capture raw frames and converts them directly into OpenCV BGR8 format using C-memory management (`ctypes`).
* **Shape & Color Detection:** Identifies objects using targeted HSV color ranges (tuned to detect red, blue, and even pale yellow wooden blocks) alongside contour geometry to classify squares and circles.
* **Socket-Based PLC Control:** Communicates directly with the PLC via standard TCP sockets over the local network (IP: 192.168.6.10, Port: 2023).
* **Automated Pick-and-Place:** Executes hardware movements using specific string protocols (e.g., `plc,6` for readiness handshakes, and `plc,4` / `plc,5` for suction cup control).

---

## Prerequisites & Installation

To run this pipeline, you must configure both the Hikrobot drivers and your Python environment.

* The Hikrobot MVS SDK must be downloaded and installed from the official Hikrobot website.
* The `MvImport` wrapper folder (found in the SDK's `Development/Samples/Python/MvImport` directory) must be copied and placed directly next to the scripts in this repository.
* You must install the required Python libraries by running `pip install opencv-python numpy`.
* Ensure your PC's network adapter is on the same subnet as the PLC (e.g., `192.168.6.X`, where X is any number except 10).

---

## Hardware Calibration (`calibrate_by_commanding_robot.py`)

Because the camera operates in pixel coordinates and the robot operates in physical millimeters, you must calibrate the system before running a live sorting sequence.

* Run `calibrate_by_commanding_robot.py`.
* This script commands the robotic arm to move to known X/Y coordinates well within the robot's safe reachable range.
* When prompted in the OpenCV live window, manually click the exact pixel where the tool tip / suction cup appears.
* The script calculates an affine transform matrix linking the camera pixels to the robot's physical workspace.
* **SAFETY WARNING:** This script moves the real physical robot. Clear the workspace of obstructions and always keep a hand on the physical Emergency Stop (E-Stop) button.

---

## Running the Sorting Pipeline (`feature_detection.py`)

Once calibration is complete and your matrix is hard-coded into the pipeline, you can begin sorting.

* Run `feature_detection.py` to initiate the camera connection and open the live OpenCV video feed.
* The script runs in a continuous free-run mode, drawing bounding boxes and labels on detected objects in real-time.
* To trigger the sorting routine, press the **'t'** (or **'x'**) key on your keyboard.
* Triggering the routine will freeze the live video, send the target coordinates to the PLC, execute the physical pick-and-place sequence for every detected object, and resume the video once finished.

**Keyboard Controls:**

* **'t'** : Trigger the PLC sorting sequence.
* **'s'** : Save a snapshot of the current video frame to your local drive.
* **'q'** : Safely break the loop, release the camera hardware, close the device connection, and exit the program.