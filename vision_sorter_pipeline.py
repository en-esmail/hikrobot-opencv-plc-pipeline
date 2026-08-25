"""
vision_sorter_pipeline.py

Hikrobot MVS SDK + OpenCV live acquisition, local shape/color detection,
and PLC-based pick-and-place sorting over a socket.

Requirements:
    - Hikrobot MVS SDK installed (use the official installer)
    - Copy the SDK's MvImport folder next to this script or set MVIMPORT_DIR
    - pip install opencv-python numpy

Usage:
    python vision_sorter_pipeline.py
    Keys: 'q' to quit, 's' to save a snapshot, 't' to trigger sorting

Calibration (pixel -> robot):
    This detector reports pixel coordinates (and a pixel-space rotation for
    squares). The robot/PLC requires coordinates in the robot base frame
    (millimeters) and a grab angle in degrees — these are different units.

    To operate the robot safely, collect 3–4 correspondences (pixel_x, pixel_y
    -> robot_X, robot_Y), then compute an affine transform (cv2.getAffineTransform
    for 3 points or cv2.estimateAffine2D for 4+ points). Replace the placeholder
    pixel_to_robot() with the fitted transform matrix before enabling real picks.

Until calibrated, pixel_to_robot() contains a simple identity-ish placeholder
so the pipeline can be exercised end-to-end without driving the robot.
"""

import os
import sys
import time
import socket
import logging
from ctypes import byref, memset, sizeof, cast, POINTER, c_ubyte

import cv2
import numpy as np

# Import custom modules
from exceptions import (
    HikrobotError, PLCError, PLCConnectionError, PLCTimeoutError,
    PLCInvalidResponseError, PLCOperationError, FrameConversionError
)
from logging_config import setup_logging, get_logger
from health_checker import SystemHealthChecker

# Setup logging
logger = setup_logging()

# ---------------------------------------------------------------------------
# 0. Make sure Windows can find the runtime DLLs, then import the SDK
# ---------------------------------------------------------------------------
if sys.platform.startswith("win"):
    RUNTIME_DIR = os.environ.get(
        "MVS_RUNTIME_DIR",
        r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
    )
    if os.path.isdir(RUNTIME_DIR):
        os.environ["PATH"] = RUNTIME_DIR + os.pathsep + os.environ["PATH"]

MVIMPORT_DIR = os.environ.get(
    "MVIMPORT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport"),
)
if not os.path.isdir(MVIMPORT_DIR):
    raise FileNotFoundError(
        f"MvImport folder not found at: {MVIMPORT_DIR}\n"
        "Either move this script so MvImport is in the same folder, "
        "or set the MVIMPORT_DIR variable/env var to the correct path."
    )
sys.path.append(MVIMPORT_DIR)

try:
    from MvCameraControl_class import *  # noqa: F401,F403  (SDK-provided module)
except ImportError as e:
    raise ImportError(
        "Could not import MvCameraControl_class. Make sure the MvImport folder "
        "from the MVS SDK's Samples/Python directory sits next to this script."
    ) from e


# ---------------------------------------------------------------------------
# 1. Helper: convert whatever pixel format the camera gives us into BGR8
#    using the SDK's own converter (robust — no manual Bayer guessing needed)
# ---------------------------------------------------------------------------
def frame_to_bgr(cam, stFrame):
    """Convert an MV_FRAME_OUT payload to a BGR numpy array using the SDK."""
    try:
        stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(stConvertParam), 0, sizeof(stConvertParam))

        stConvertParam.nWidth = stFrame.stFrameInfo.nWidth
        stConvertParam.nHeight = stFrame.stFrameInfo.nHeight
        stConvertParam.pSrcData = stFrame.pBufAddr
        stConvertParam.nSrcDataLen = stFrame.stFrameInfo.nFrameLen
        stConvertParam.enSrcPixelType = stFrame.stFrameInfo.enPixelType
        stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed

        dst_size = stFrame.stFrameInfo.nWidth * stFrame.stFrameInfo.nHeight * 3
        dst_buf = (c_ubyte * dst_size)()
        stConvertParam.pDstBuffer = cast(dst_buf, POINTER(c_ubyte))
        stConvertParam.nDstBufferSize = dst_size

        ret = cam.MV_CC_ConvertPixelType(stConvertParam)
        if ret != 0:
            raise FrameConversionError(f"Pixel conversion failed. Error code: 0x{ret:x}")

        img = np.frombuffer(dst_buf, dtype=np.uint8, count=dst_size)
        img = img.reshape(stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth, 3)
        logger.debug(f"Frame converted successfully: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"Frame conversion error: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# 2. Shape + color detection
# ---------------------------------------------------------------------------

# HSV ranges for common colors. Tune these to your lighting / objects.
# Red wraps around the hue circle, so it needs two ranges.
COLOR_RANGES = {
    "Red": [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
    "Yellow": [((15, 40, 50), (45, 255, 255))],  # Expanded to catch pale wood tones
    "Blue": [((85, 50, 50), (130, 255, 255))],
}

MIN_CONTOUR_AREA = 500  # ignore tiny noise blobs
DRAW_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 255)  # red so it stands out on white backgrounds


def classify_shape(contour):
    """Classify a contour's shape, returning only Square, Circle, or None."""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    sides = len(approx)

    if sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.80 <= aspect_ratio <= 1.20:
            return "Square"

    area = cv2.contourArea(contour)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.75:
            return "Circle"

    return None


def classify_color(hsv_roi, mask_roi):
    """Return the color name whose HSV range covers the most pixels in the ROI."""
    best_color, best_count = "Unknown", 0
    for name, ranges in COLOR_RANGES.items():
        total = 0
        for lower, upper in ranges:
            m = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
            m = cv2.bitwise_and(m, mask_roi)
            total += cv2.countNonZero(m)
        if total > best_count:
            best_color, best_count = name, total
    return best_color


def _contour_angle(contour, shape):
    """Grab angle for the robot, in degrees. Squares use the minAreaRect
    rotation; circles have no meaningful rotation so we report 0."""
    if shape != "Square":
        return 0.0
    rect = cv2.minAreaRect(contour)
    return rect[-1]


def detect_objects(frame_bgr):
    """Run shape/color detection and return a list of clean dicts:
        {"shape", "color", "x", "y", "angle"}
    where x/y are the pixel-space centroid (bounding-box center) of the
    object. This is what feeds the sorting sequence below.
    """
    try:
        blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue

            shape = classify_shape(contour)
            if shape not in ["Square", "Circle"]:
                continue

            obj_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(obj_mask, [contour], -1, 255, thickness=cv2.FILLED)
            color = classify_color(hsv, obj_mask)
            if color not in ["Red", "Blue", "Yellow"]:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w / 2.0, y + h / 2.0
            angle = _contour_angle(contour, shape)

            results.append({
                "shape": shape,
                "color": color,
                "x": cx,
                "y": cy,
                "angle": angle,
            })
        
        logger.debug(f"Detected {len(results)} objects in frame")
        return results
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        raise


def process_frame(frame_bgr):
    """
    Detect objects, draw bounding boxes + labels for the live view, and
    print results to the terminal. Uses detect_objects() internally so
    the live view and the sorting trigger never disagree with each other.
    """
    detections = detect_objects(frame_bgr)

    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections_this_frame = []
    det_idx = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue
        shape = classify_shape(contour)
        if shape not in ["Square", "Circle"]:
            continue
        x, y, w, h = cv2.boundingRect(contour)

        if det_idx < len(detections):
            det = detections[det_idx]
            det_idx += 1
            color = det["color"]
        else:
            continue

        detections_this_frame.append(f"{color} {shape} at X:{x} Y:{y}")

        cv2.drawContours(frame_bgr, [contour], -1, DRAW_COLOR, 2)
        label = f"{color} {shape}"
        cv2.rectangle(frame_bgr, (x, y - 25), (x + len(label) * 12, y), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame_bgr, label, (x + 2, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

    if detections_this_frame:
        print(f"[{time.strftime('%H:%M:%S')}] Detected: " + " | ".join(detections_this_frame))

    return frame_bgr


# ---------------------------------------------------------------------------
# 3. Camera-pixel -> robot-frame calibration
# ---------------------------------------------------------------------------
# Fitted from calibrate_all_in_one.py — maps camera pixel (px, py) to
# robot (X, Y) in mm. Re-run calibration and replace this matrix if the
# camera, lens, or mount ever changes.
PIXEL_TO_ROBOT_MATRIX = np.array([
    [-0.001169, -0.086443, 94.633716],
    [-0.085307, 0.001197, 89.196052],
])


def pixel_to_robot(px, py):
    """Convert a pixel (px, py) from detect_objects() into robot X/Y (mm)
    using the fitted calibration matrix above."""
    vec = np.array([px, py, 1.0])
    robot_x, robot_y = PIXEL_TO_ROBOT_MATRIX @ vec
    return robot_x, robot_y


# ---------------------------------------------------------------------------
# 4. Which detections count as "target" vs "reject"
# ---------------------------------------------------------------------------
def is_target_object(shape, color):
    """Mirrors the reference script's `shape == '10' and color == 'yellow'`
    rule (square + yellow -> goes to the "good" grid). Edit this to match
    whatever sorting rule you actually need.
    """
    return shape == "Square" and color == "Yellow"


# ---------------------------------------------------------------------------
# 5. PLC / robot controller (pick-and-place sorting logic)
# ---------------------------------------------------------------------------
class PLCController:
    """PLC robot controller for pick-and-place sorting with validation."""
    
    BUFSIZE = 2048
    MAX_RESPONSE_LENGTH = 100  # Max expected length of PLC response
    VALID_RESPONSE_PATTERN = r"^\d+$"  # PLC responses should be numeric

    def __init__(self, ip="192.168.6.10", port=2023,
                 z_above="-30", z_grab="-84"):
        self.ip = ip
        self.port = port
        self.z = z_above   # height above grab/place point
        self.H = z_grab    # actual grab/place point height
        self.client = None
        self.logger = get_logger()

        # bin-fill counters, persist across calls (like `a` and `b`
        # in the reference script)
        self.count_target = 0
        self.count_reject = 0
        
        self.logger.info(f"PLCController initialized: {ip}:{port}, z_above={z_above}, z_grab={z_grab}")

    # --- Response Validation ---
    def _validate_response(self, response: str) -> bool:
        """Validate PLC response format and content.
        
        Args:
            response: Raw response from PLC
            
        Returns:
            True if response is valid, False otherwise
            
        Raises:
            PLCInvalidResponseError: If response is invalid
        """
        if not response:
            self.logger.error("PLC response is empty")
            raise PLCInvalidResponseError("PLC response is empty")
        
        response_stripped = response.strip()
        
        # Check response length
        if len(response_stripped) > self.MAX_RESPONSE_LENGTH:
            self.logger.error(f"PLC response too long: {len(response_stripped)} chars (max {self.MAX_RESPONSE_LENGTH})")
            raise PLCInvalidResponseError(f"Response too long: {len(response_stripped)} chars")
        
        # Check response contains only expected characters (digits)
        if not response_stripped.isdigit():
            self.logger.error(f"PLC response contains invalid characters: {response_stripped!r}")
            raise PLCInvalidResponseError(f"Response contains invalid characters: {response_stripped!r}")
        
        self.logger.debug(f"PLC response validated: {response_stripped!r}")
        return True

    # -- low level -----------------------------------------------------
    def connect(self):
        """Establish socket connection to PLC."""
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((self.ip, self.port))
            self.logger.info(f"Connected to PLC at {self.ip}:{self.port}")
        except socket.timeout as e:
            self.logger.error(f"Connection timeout to PLC at {self.ip}:{self.port}")
            raise PLCTimeoutError(f"Connection timeout: {e}") from e
        except socket.error as e:
            self.logger.error(f"Failed to connect to PLC at {self.ip}:{self.port}: {e}")
            raise PLCConnectionError(f"Connection failed: {e}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to PLC: {e}", exc_info=True)
            raise PLCConnectionError(f"Unexpected error: {e}") from e

    def close(self):
        """Close connection to PLC."""
        if self.client is not None:
            try:
                self.client.close()
                self.logger.info("PLC connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing PLC connection: {e}")
            finally:
                self.client = None

    def _task(self, msg):
        """Send command to PLC and receive response with validation.
        
        Args:
            msg: Command message to send
            
        Returns:
            Response string from PLC
            
        Raises:
            PLCConnectionError: If connection is lost
            PLCTimeoutError: If no response is received
            PLCInvalidResponseError: If response is invalid
        """
        try:
            self.logger.debug(f"Sending PLC command: {msg!r}")
            self.client.send(msg.encode("utf-8"))
            recv_data = self.client.recv(self.BUFSIZE).decode("gbk")
            self._validate_response(recv_data)
            self.logger.debug(f"PLC response: {recv_data!r}")
            return recv_data
        except socket.timeout as e:
            self.logger.error(f"PLC communication timeout: {e}")
            raise PLCTimeoutError(f"No response from PLC: {e}") from e
        except socket.error as e:
            self.logger.error(f"PLC socket error: {e}")
            raise PLCConnectionError(f"Connection lost: {e}") from e
        except (PLCInvalidResponseError, PLCTimeoutError, PLCConnectionError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in PLC communication: {e}", exc_info=True)
            raise PLCOperationError(f"Communication error: {e}") from e

    def _handshake(self):
        """Send handshake command and validate response."""
        try:
            response = self._task("plc,6")
            if response.strip() != "1":
                self.logger.warning(f"Handshake returned non-ready status: {response!r}")
            return response
        except Exception as e:
            self.logger.error(f"Handshake failed: {e}")
            raise

    def _move(self, x, y, z, angle):
        """Send move command and validate response.
        
        Args:
            x, y, z: Position coordinates
            angle: Rotation angle in degrees
        """
        try:
            response = self._task(f"plc,{x},{y},{z},{angle},7")
            if response.strip() != "1":
                self.logger.warning(f"Move returned non-ready status: {response!r}")
            return response
        except Exception as e:
            self.logger.error(f"Move command failed at ({x}, {y}, {z}, {angle}): {e}")
            raise

    # -- shared motions --------------------------------------------------
    def go_to_photo_point(self, x="20", y="85", z="20", angle="0"):
        """Move to the fixed photo point, mirrors the reference script's
        'plc,20,85,20,0,7' call. Returns the PLC in-position reply so the
        caller can check it's '1' before triggering the camera, exactly
        like the reference script does.
        """
        self._handshake()
        return self._move(x, y, z, angle)

    def go_home(self):
        self._handshake()
        self._task("plc,0,0,0,0,7")

    def _grab(self, x, y, angle):
        """Above grab point -> down to grab height -> suction ON -> back up.
        Same 4-step sequence as sorting() in the reference script.
        """
        self._handshake()
        self._move(x, y, self.z, angle)          # above grab point

        self._handshake()
        self._move(x, y, self.H, angle)          # grab point

        self._task("plc,4")                       # open suction cup (grab)

        self._handshake()
        self._move(x, y, self.z, angle)          # above grab point

    def _place(self, x, y, angle="0"):
        """Above place point -> down to place height -> suction OFF -> back up."""
        self._handshake()
        self._move(x, y, self.z, angle)          # above placement point

        self._handshake()
        self._move(x, y, self.H, angle)          # placement point

        self._task("plc,5")                       # close suction cup (release)

        self._handshake()
        self._move(x, y, self.z, angle)          # above placement point

    # -- grid math, ported directly from the reference script -----------
    @staticmethod
    def _grid_slot(count):
        """Same row/column/height math as the reference script's
        `if a < 6: ... else: ...` blocks (2 layers of a 2x3 grid, 6 slots
        per layer)."""
        if count < 6:
            row = count // 3
            column = count % 3
            height = count // 6
        else:
            row = (count - 6) // 3
            column = (count - 6) % 3
            height = count // 6
        return row, column, height

    def _place_target(self):
        """Target ("good") bin placement — mirrors the "soybean" placement
        block in the reference script."""
        row, column, _height = self._grid_slot(self.count_target)
        place_x = -101 - row * 45
        place_y = 55.5 - column * 45
        self._place(str(place_x), str(place_y), "0")
        self.count_target += 1

    def _place_reject(self):
        """Reject ("foreign object") bin placement — mirrors the reference
        script's foreign-object placement block."""
        row, column, _height = self._grid_slot(self.count_reject)
        place_x = 160.5 - row * 45
        place_y = 60.5 - column * 45
        self._place(str(place_x), str(place_y), "0")
        self.count_reject += 1

    # -- top level --------------------------------------------------------
    def process_detections(self, detections, use_robot_coords=False):
        """Grab and sort every detection in `detections`.

        detections: list of dicts, one per object found by detect_objects(),
            each shaped like:
                {
                    "shape": "Square" | "Circle",
                    "color": "Red" | "Yellow" | "Blue",
                    "x": <pixel or robot X>,
                    "y": <pixel or robot Y>,
                    "angle": <grab angle, degrees, default 0>,
                }

        use_robot_coords: if False (default), "x"/"y" in each detection are
            treated as raw camera pixels and passed through pixel_to_robot()
            first. Set to True once upstream code already supplies
            calibrated robot coordinates.
        """
        if not detections:
            self.logger.warning("process_detections called with empty detection list")
            return
        
        self.logger.info(f"Processing {len(detections)} detections")
        
        for i, det in enumerate(detections):
            try:
                shape = det["shape"]
                color = det["color"]
                angle = str(det.get("angle", 0))

                if use_robot_coords:
                    robot_x, robot_y = det["x"], det["y"]
                else:
                    robot_x, robot_y = pixel_to_robot(det["x"], det["y"])

                x_str, y_str = str(robot_x), str(robot_y)

                self.logger.info(f"[{i + 1}/{len(detections)}] {color} {shape} "
                                f"-> grabbing at ({x_str}, {y_str}), angle={angle}°")
                print(f"[{i + 1}/{len(detections)}] {color} {shape} "
                      f"-> grabbing at ({x_str}, {y_str})")

                self._grab(x_str, y_str, angle)

                if is_target_object(shape, color):
                    self.logger.debug(f"Object is target, placing in target bin")
                    self._place_target()
                else:
                    self.logger.debug(f"Object is reject, placing in reject bin")
                    self._place_reject()
                
                self.logger.info(f"Successfully processed object {i + 1}/{len(detections)}")
            except (PLCError, Exception) as e:
                self.logger.error(f"Failed to process detection {i + 1}/{len(detections)}: {e}", exc_info=True)
                raise


def run_sorting(detections, ip="192.168.6.10", port=2023,
                 z_above="-30", z_grab="-84", go_home_after=True):
    """One-call helper: connect, run through every detection, go home, close.

    Called from the live-view loop below when 't' is pressed, the same
    role the reference script's "vm,0" trigger used to play.
    """
    logger = get_logger()
    
    if not detections:
        logger.info("No objects to sort, skipping.")
        print("No objects to sort, skipping.")
        return

    plc = PLCController(ip=ip, port=port, z_above=z_above, z_grab=z_grab)
    
    try:
        logger.info(f"Starting sorting sequence with {len(detections)} objects")
        plc.connect()
        plc.process_detections(detections)
        if go_home_after:
            logger.info("Returning to home position")
            plc.go_to_photo_point()
        logger.info("Sorting task completed successfully")
        print("Sorting task completed.")
    except PLCConnectionError as e:
        logger.error(f"PLC connection failed: {e}")
        print(f"ERROR: Failed to connect to PLC: {e}")
    except PLCTimeoutError as e:
        logger.error(f"PLC communication timeout: {e}")
        print(f"ERROR: PLC timeout: {e}")
    except PLCInvalidResponseError as e:
        logger.error(f"Invalid PLC response: {e}")
        print(f"ERROR: Invalid PLC response: {e}")
    except PLCOperationError as e:
        logger.error(f"PLC operation failed: {e}")
        print(f"ERROR: PLC operation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during sorting: {e}", exc_info=True)
        print(f"ERROR: Unexpected error: {e}")
    finally:
        plc.close()


# ---------------------------------------------------------------------------
# 6. Main capture loop
# ---------------------------------------------------------------------------
def main():
    SDKVersion = MvCamera.MV_CC_GetSDKVersion()
    print(f"MVS SDK Version: 0x{SDKVersion:x}")

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        raise RuntimeError(f"Enum devices failed! Error code: 0x{ret:x}")
    if deviceList.nDeviceNum == 0:
        raise RuntimeError("No Hikrobot camera found. Check power/cable/network.")

    print(f"Found {deviceList.nDeviceNum} device(s):")
    for i in range(deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            ip = mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp
            ip_str = f"{(ip >> 24) & 0xff}.{(ip >> 16) & 0xff}.{(ip >> 8) & 0xff}.{ip & 0xff}"
            print(f"  [{i}] GigE camera - IP: {ip_str}")
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print(f"  [{i}] USB camera")

    device_index = 0
    stDeviceInfo = cast(deviceList.pDeviceInfo[device_index], POINTER(MV_CC_DEVICE_INFO)).contents

    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(stDeviceInfo)
    if ret != 0:
        raise RuntimeError(f"Create handle failed! Error code: 0x{ret:x}")

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        cam.MV_CC_DestroyHandle()
        raise RuntimeError(f"Open device failed! Error code: 0x{ret:x}")

    if stDeviceInfo.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if nPacketSize > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)

    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    cam.MV_CC_SetEnumValue("ExposureAuto", 0)
    cam.MV_CC_SetFloatValue("ExposureTime", 10000.0)
    cam.MV_CC_SetFloatValue("Gain", 5.0)

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        raise RuntimeError(f"Start grabbing failed! Error code: 0x{ret:x}")

    # Perform health checks before starting
    logger.info("Performing system health checks...")
    health_checker = SystemHealthChecker()
    all_healthy, check_results = health_checker.check_all()
    health_checker.print_report()
    
    if not all_healthy:
        logger.warning("Some health checks failed. Proceeding with caution.")
        print("WARNING: Some system components may not be ready.")
    else:
        logger.info("All health checks passed. System is ready.")

    print("Moving robot to photo point before starting...")
    plc = PLCController()
    plc.connect()
    try:
        plc.go_to_photo_point()
    finally:
        plc.close()

    print("Streaming started. Press 'q' to quit, 's' to save a snapshot, "
          "'t' to trigger sorting.")

    snapshot_count = 0

    cv2.namedWindow("Hikrobot Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hikrobot Camera", 800, 600)

    latest_detections = []

    try:
        while True:
            stFrame = MV_FRAME_OUT()
            memset(byref(stFrame), 0, sizeof(stFrame))

            ret = cam.MV_CC_GetImageBuffer(stFrame, 1000)
            if ret != 0:
                time.sleep(0.005)
                continue

            try:
                frame_bgr = frame_to_bgr(cam, stFrame)
            finally:
                cam.MV_CC_FreeImageBuffer(stFrame)

            latest_detections = detect_objects(frame_bgr)
            frame_bgr = process_frame(frame_bgr)

            cv2.imshow("Hikrobot Camera", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                snapshot_count += 1
                fname = f"snapshot_{snapshot_count:03d}.png"
                cv2.imwrite(fname, frame_bgr)
                print(f"Saved {fname}")
            elif key == ord('t'):
                # Same role as the reference script's "vm,0" trigger:
                # take whatever is currently detected and run the
                # pick-and-place sorting sequence over it.
                print(f"Triggering sort on {len(latest_detections)} object(s)...")
                run_sorting(latest_detections)

    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("Camera closed cleanly.")


if __name__ == "__main__":
    main()
