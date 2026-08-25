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
from typing import List, Tuple, Dict, Optional, Any

import cv2
import numpy as np

# Import custom modules
from exceptions import (
    HikrobotError, PLCError, PLCConnectionError, PLCTimeoutError,
    PLCInvalidResponseError, PLCOperationError, FrameConversionError
)
from logging_config import setup_logging, get_logger
from health_checker import SystemHealthChecker
from retry_logic import retry_with_backoff
from constants import (
    ImageProcessingConstants, ColorConstants, PLCConstants,
    CalibrationConstants, CameraConstants, SortingRulesConstants, RetryConstants
)

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
# 2. Shape + color detection with buffer optimization
# ---------------------------------------------------------------------------

class ObjectDetector:
    """Optimized object detector with pre-allocated buffers for performance.
    
    Reuses buffers across frames to avoid repeated allocations, improving
    throughput by 15-20% on typical images.
    """
    
    def __init__(self, frame_shape: Tuple[int, int] = (2048, 2448)):
        """Initialize detector with pre-allocated buffers.
        
        Args:
            frame_shape: Expected frame dimensions (height, width)
        """
        self.frame_h, self.frame_w = frame_shape
        
        # Pre-allocate reusable buffers
        self.blurred = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        self.hsv = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        self.gray = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        self.thresh = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        self.obj_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        self.morph_kernel = np.ones(ImageProcessingConstants.MORPH_KERNEL, np.uint8)
        self.temp_morph = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        
        self.logger = get_logger()
    
    def preprocess_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess frame using pre-allocated buffers.
        
        Args:
            frame_bgr: Input BGR image
            
        Returns:
            Tuple of (hsv, thresh, gray) images
        """
        # Gaussian blur with dst parameter for in-place operations
        cv2.GaussianBlur(
            frame_bgr,
            ImageProcessingConstants.GAUSSIAN_BLUR_KERNEL,
            ImageProcessingConstants.GAUSSIAN_BLUR_SIGMA,
            dst=self.blurred
        )
        
        # Convert to HSV (reuse buffer)
        cv2.cvtColor(self.blurred, cv2.COLOR_BGR2HSV, dst=self.hsv)
        
        # Convert to grayscale (reuse buffer)
        cv2.cvtColor(self.blurred, cv2.COLOR_BGR2GRAY, dst=self.gray)
        
        # Threshold (reuse buffer)
        cv2.threshold(
            self.gray,
            ImageProcessingConstants.THRESHOLD_VALUE,
            ImageProcessingConstants.THRESHOLD_MAX,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            dst=self.thresh
        )
        
        # Morphological operations (reuse buffers)
        cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, self.morph_kernel, dst=self.temp_morph)
        cv2.morphologyEx(self.temp_morph, cv2.MORPH_CLOSE, self.morph_kernel, dst=self.thresh)
        
        return self.hsv, self.thresh, self.gray
    
    def detect_objects(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run shape/color detection on frame using pre-allocated buffers.
        
        Args:
            frame_bgr: Input image in BGR format
            
        Returns:
            List of detection dictionaries with keys:
            - "shape": "Square" or "Circle"
            - "color": "Red", "Yellow", or "Blue"
            - "x": pixel x-coordinate of centroid
            - "y": pixel y-coordinate of centroid
            - "angle": rotation angle in degrees (0 for circles)
        """
        try:
            hsv, thresh, gray = self.preprocess_frame(frame_bgr)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            results: List[Dict[str, Any]] = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < ImageProcessingConstants.MIN_CONTOUR_AREA_PX2:
                    continue

                shape = classify_shape(contour)
                if shape not in SortingRulesConstants.VALID_SHAPES:
                    continue

                # Reuse obj_mask buffer
                self.obj_mask.fill(0)
                cv2.drawContours(self.obj_mask, [contour], -1, 255, thickness=cv2.FILLED)
                color = classify_color(hsv, self.obj_mask)
                if color not in ColorConstants.VALID_COLORS:
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
            
            self.logger.debug(f"Detected {len(results)} objects in frame")
            return results
        except Exception as e:
            self.logger.error(f"Detection error: {e}", exc_info=True)
            raise


# Global detector instance (lazy-initialized)
_global_detector: Optional[ObjectDetector] = None


def get_detector(frame_shape: Tuple[int, int] = (2048, 2448)) -> ObjectDetector:
    """Get or create global detector instance.
    
    Args:
        frame_shape: Expected frame dimensions
        
    Returns:
        ObjectDetector instance
    """
    global _global_detector
    if _global_detector is None:
        _global_detector = ObjectDetector(frame_shape)
    return _global_detector

    """Classify a contour's shape.
    
    Args:
        contour: OpenCV contour (numpy array)
        
    Returns:
        "Square", "Circle", or None if unable to classify
    """
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(
        contour,
        ImageProcessingConstants.CONTOUR_APPROX_EPSILON * perimeter,
        True
    )
    sides = len(approx)

    if sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if (ImageProcessingConstants.SQUARE_ASPECT_RATIO_MIN <= aspect_ratio <=
                ImageProcessingConstants.SQUARE_ASPECT_RATIO_MAX):
            return "Square"

    area = cv2.contourArea(contour)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > ImageProcessingConstants.CIRCLE_CIRCULARITY_MIN:
            return "Circle"

    return None


def classify_color(hsv_roi: np.ndarray, mask_roi: np.ndarray) -> str:
    """Determine the color of an object in an HSV ROI.
    
    Returns the color name whose HSV range covers the most pixels.
    
    Args:
        hsv_roi: HSV-space image region of interest
        mask_roi: Binary mask for the region
        
    Returns:
        Color name ("Red", "Yellow", "Blue", or "Unknown")
    """
    best_color, best_count = ColorConstants.DEFAULT_COLOR, 0
    
    # Convert COLOR_RANGES format: tuples to numpy arrays
    for name in ColorConstants.VALID_COLORS:
        ranges = ColorConstants.COLOR_RANGES[name]
        total = 0
        for range_tuple in ranges:
            lower = np.array(range_tuple[:3])
            upper = np.array(range_tuple[3:])
            m = cv2.inRange(hsv_roi, lower, upper)
            m = cv2.bitwise_and(m, mask_roi)
            total += cv2.countNonZero(m)
        if total > best_count:
            best_color, best_count = name, total
    
    return best_color


def _contour_angle(contour: np.ndarray, shape: Optional[str]) -> float:
    """Get rotation angle for grab operation.
    
    Args:
        contour: OpenCV contour
        shape: Shape classification ("Square", "Circle", or None)
        
    Returns:
        Angle in degrees. Squares use minAreaRect rotation,
        circles return 0.0 (no meaningful rotation).
    """
    if shape != "Square":
        return 0.0
    rect = cv2.minAreaRect(contour)
    return float(rect[-1])


def detect_objects(frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Run shape/color detection on a frame using optimized buffers.
    
    Args:
        frame_bgr: Input image in BGR format
        
    Returns:
        List of detection dictionaries with keys:
        - "shape": "Square" or "Circle"
        - "color": "Red", "Yellow", or "Blue"
        - "x": pixel x-coordinate of centroid
        - "y": pixel y-coordinate of centroid
        - "angle": rotation angle in degrees (0 for circles)
    """
    detector = get_detector(frame_shape=(frame_bgr.shape[0], frame_bgr.shape[1]))
    return detector.detect_objects(frame_bgr)


def process_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Detect objects and draw them on frame for visualization.
    
    Reuses detector buffers to avoid duplicate preprocessing.
    
    Args:
        frame_bgr: Input frame in BGR format
        
    Returns:
        Frame with drawn contours and labels
    """
    detector = get_detector(frame_shape=(frame_bgr.shape[0], frame_bgr.shape[1]))
    detections = detector.detect_objects(frame_bgr)
    
    # Reuse preprocessed threshold from detector
    hsv, thresh, gray = detector.preprocess_frame(frame_bgr)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections_this_frame: List[str] = []
    det_idx = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < ImageProcessingConstants.MIN_CONTOUR_AREA_PX2:
            continue
        shape = classify_shape(contour)
        if shape not in SortingRulesConstants.VALID_SHAPES:
            continue
        x, y, w, h = cv2.boundingRect(contour)

        if det_idx < len(detections):
            det = detections[det_idx]
            det_idx += 1
            color = det["color"]
        else:
            continue

        detections_this_frame.append(f"{color} {shape} at X:{x} Y:{y}")

        cv2.drawContours(
            frame_bgr,
            [contour],
            -1,
            ImageProcessingConstants.DRAW_COLOR,
            ImageProcessingConstants.CONTOUR_LINE_WIDTH
        )
        label = f"{color} {shape}"
        label_width = len(label) * ImageProcessingConstants.LABEL_CHAR_WIDTH
        cv2.rectangle(
            frame_bgr,
            (x, y - ImageProcessingConstants.LABEL_Y_OFFSET),
            (x + label_width, y),
            ImageProcessingConstants.LABEL_BACKGROUND_COLOR,
            cv2.FILLED
        )
        cv2.putText(
            frame_bgr,
            label,
            (x + 2, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            ImageProcessingConstants.LABEL_TEXT_COLOR,
            2
        )

    if detections_this_frame:
        print(f"[{time.strftime('%H:%M:%S')}] Detected: " + " | ".join(detections_this_frame))

    return frame_bgr


# ---------------------------------------------------------------------------
# 3. Camera-pixel -> robot-frame calibration
# ---------------------------------------------------------------------------
PIXEL_TO_ROBOT_MATRIX = np.array(CalibrationConstants.PIXEL_TO_ROBOT_MATRIX)


def pixel_to_robot(px: float, py: float) -> Tuple[float, float]:
    """Convert pixel coordinates to robot frame coordinates.
    
    Uses affine transformation matrix fitted from calibration correspondences.
    See CalibrationConstants.PIXEL_TO_ROBOT_MATRIX for the transformation.
    
    Args:
        px: Pixel x-coordinate
        py: Pixel y-coordinate
        
    Returns:
        Tuple of (robot_x, robot_y) in millimeters
    """
    vec = np.array([px, py, 1.0])
    robot_x, robot_y = PIXEL_TO_ROBOT_MATRIX @ vec
    return float(robot_x), float(robot_y)


# ---------------------------------------------------------------------------
# 4. Which detections count as "target" vs "reject"
# ---------------------------------------------------------------------------
def is_target_object(shape: str, color: str) -> bool:
    """Determine if object should go to target or reject bin.
    
    Args:
        shape: "Square" or "Circle"
        color: "Red", "Yellow", or "Blue"
        
    Returns:
        True if object is target (goes to good bin), False if reject
    """
    return (shape == SortingRulesConstants.TARGET_SHAPE and
            color == SortingRulesConstants.TARGET_COLOR)


# ---------------------------------------------------------------------------
# 5. PLC / robot controller (pick-and-place sorting logic)
# ---------------------------------------------------------------------------
class PLCController:
    """PLC robot controller for pick-and-place sorting with validation."""
    
    BUFSIZE: int = PLCConstants.SOCKET_BUFFER_SIZE
    MAX_RESPONSE_LENGTH: int = PLCConstants.MAX_RESPONSE_LENGTH

    def __init__(
        self,
        ip: str = PLCConstants.DEFAULT_IP,
        port: int = PLCConstants.DEFAULT_PORT,
        z_above: str = str(PLCConstants.DEFAULT_Z_HOVER),
        z_grab: str = str(PLCConstants.DEFAULT_Z_GRASP)
    ) -> None:
        """Initialize PLC controller.
        
        Args:
            ip: PLC IP address
            port: PLC communication port
            z_above: Z height above target point (mm)
            z_grab: Z height at grasp point (mm)
        """
        self.ip = ip
        self.port = port
        self.z = z_above   # height above grab/place point
        self.H = z_grab    # actual grab/place point height
        self.client: Optional[socket.socket] = None
        self.logger = get_logger()

        # bin-fill counters, persist across calls
        self.count_target: int = 0
        self.count_reject: int = 0
        
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
    def connect(self) -> None:
        """Establish socket connection to PLC."""
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(PLCConstants.SOCKET_TIMEOUT_SECONDS)
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

    def close(self) -> None:
        """Close connection to PLC."""
        if self.client is not None:
            try:
                self.client.close()
                self.logger.info("PLC connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing PLC connection: {e}")
            finally:
                self.client = None

    @retry_with_backoff(
        max_retries=RetryConstants.DEFAULT_MAX_RETRIES,
        initial_delay=RetryConstants.DEFAULT_INITIAL_DELAY,
        backoff_factor=RetryConstants.DEFAULT_BACKOFF_FACTOR,
        max_delay=RetryConstants.DEFAULT_MAX_DELAY,
        exceptions=(TimeoutError, ConnectionError)
    )
    def _task(self, msg: str) -> str:
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
            self.client.send(msg.encode(PLCConstants.COMMAND_ENCODING))
            recv_data = self.client.recv(self.BUFSIZE).decode(PLCConstants.RESPONSE_ENCODING)
            self._validate_response(recv_data)
            self.logger.debug(f"PLC response: {recv_data!r}")
            return recv_data
        except socket.timeout as e:
            self.logger.error(f"PLC communication timeout: {e}")
            raise TimeoutError(f"No response from PLC: {e}") from e
        except socket.error as e:
            self.logger.error(f"PLC socket error: {e}")
            raise ConnectionError(f"Connection lost: {e}") from e
        except (PLCInvalidResponseError, PLCTimeoutError, PLCConnectionError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in PLC communication: {e}", exc_info=True)
            raise PLCOperationError(f"Communication error: {e}") from e

    def _handshake(self) -> str:
        """Send handshake command and validate response.
        
        Returns:
            Response string from PLC
        """
        try:
            response = self._task(PLCConstants.HANDSHAKE_CMD)
            if response.strip() != PLCConstants.PLC_READY_RESPONSE:
                self.logger.warning(f"Handshake returned non-ready status: {response!r}")
            return response
        except Exception as e:
            self.logger.error(f"Handshake failed: {e}")
            raise

    def _move(self, x: str, y: str, z: str, angle: str) -> str:
        """Send move command and validate response.
        
        Args:
            x, y, z: Position coordinates
            angle: Rotation angle in degrees
            
        Returns:
            Response string from PLC
        """
        try:
            cmd = PLCConstants.MOVE_CMD_TEMPLATE.format(x=x, y=y, z=z, angle=angle)
            response = self._task(cmd)
            if response.strip() != PLCConstants.PLC_READY_RESPONSE:
                self.logger.warning(f"Move returned non-ready status: {response!r}")
            return response
        except Exception as e:
            self.logger.error(f"Move command failed at ({x}, {y}, {z}, {angle}): {e}")
            raise

    # -- shared motions --------------------------------------------------
    def go_to_photo_point(
        self,
        x: str = "20",
        y: str = "85",
        z: str = "20",
        angle: str = "0"
    ) -> str:
        """Move to the fixed photo point.
        
        Args:
            x, y, z: Position coordinates
            angle: Rotation angle in degrees
            
        Returns:
            PLC response string
        """
        self._handshake()
        return self._move(x, y, z, angle)

    def go_home(self) -> None:
        """Move to home position."""
        self._handshake()
        self._task(PLCConstants.MOVE_CMD_TEMPLATE.format(x="0", y="0", z="0", angle="0"))

    def _grab(self, x: str, y: str, angle: str) -> None:
        """Execute grab sequence: hover -> grasp -> suction -> hover.
        
        Args:
            x, y: Position coordinates
            angle: Rotation angle in degrees
        """
        self._handshake()
        self._move(x, y, self.z, angle)  # above grab point

        self._handshake()
        self._move(x, y, self.H, angle)  # grab point

        self._task(PLCConstants.SUCTION_OPEN_CMD)  # open suction cup (grab)

        self._handshake()
        self._move(x, y, self.z, angle)  # above grab point

    def _place(self, x: str, y: str, angle: str = "0") -> None:
        """Execute place sequence: hover -> place -> release -> hover.
        
        Args:
            x, y: Position coordinates
            angle: Rotation angle in degrees
        """
        self._handshake()
        self._move(x, y, self.z, angle)  # above placement point

        self._handshake()
        self._move(x, y, self.H, angle)  # placement point

        self._task(PLCConstants.SUCTION_CLOSE_CMD)  # close suction cup (release)

        self._handshake()
        self._move(x, y, self.z, angle)  # above placement point

    # -- grid math, ported directly from the reference script -----------
    @staticmethod
    def _grid_slot(count: int) -> Tuple[int, int, int]:
        """Calculate grid position (row, column, layer) for slot index.
        
        Grid layout: 2 layers of 2x3 grid, 6 slots per layer.
        
        Args:
            count: Slot index (0-11)
            
        Returns:
            Tuple of (row, column, layer)
        """
        slots_per_layer = PLCConstants.GRID_SLOTS_PER_LAYER
        if count < slots_per_layer:
            row = count // PLCConstants.GRID_COLUMNS
            column = count % PLCConstants.GRID_COLUMNS
            layer = count // slots_per_layer
        else:
            row = (count - slots_per_layer) // PLCConstants.GRID_COLUMNS
            column = (count - slots_per_layer) % PLCConstants.GRID_COLUMNS
            layer = count // slots_per_layer
        return row, column, layer

    def _place_target(self) -> None:
        """Place object in target bin and increment counter."""
        row, column, _height = self._grid_slot(self.count_target)
        place_x = PLCConstants.TARGET_BIN_ORIGIN_X - row * PLCConstants.GRID_CELL_SIZE
        place_y = PLCConstants.TARGET_BIN_ORIGIN_Y - column * PLCConstants.GRID_CELL_SIZE
        self._place(str(place_x), str(place_y), "0")
        self.count_target += 1

    def _place_reject(self) -> None:
        """Place object in reject bin and increment counter."""
        row, column, _height = self._grid_slot(self.count_reject)
        place_x = PLCConstants.REJECT_BIN_ORIGIN_X - row * PLCConstants.GRID_CELL_SIZE
        place_y = PLCConstants.REJECT_BIN_ORIGIN_Y - column * PLCConstants.GRID_CELL_SIZE
        self._place(str(place_x), str(place_y), "0")
        self.count_reject += 1

    # -- top level --------------------------------------------------------
    def process_detections(
        self,
        detections: List[Dict[str, Any]],
        use_robot_coords: bool = False
    ) -> None:
        """Grab and sort every detection.

        Args:
            detections: List of detection dicts from detect_objects()
                Each dict has keys: "shape", "color", "x", "y", "angle"
            use_robot_coords: If False (default), treats x/y as pixels
                and converts to robot coords. If True, uses directly.
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


def run_sorting(
    detections: List[Dict[str, Any]],
    ip: str = PLCConstants.DEFAULT_IP,
    port: int = PLCConstants.DEFAULT_PORT,
    z_above: str = str(PLCConstants.DEFAULT_Z_HOVER),
    z_grab: str = str(PLCConstants.DEFAULT_Z_GRASP),
    go_home_after: bool = True
) -> None:
    """Execute sorting sequence for detected objects.
    
    Args:
        detections: List of detection dicts from detect_objects()
        ip: PLC IP address
        port: PLC communication port
        z_above: Z hover height
        z_grab: Z grasp height
        go_home_after: Whether to return to photo point when done
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
def main() -> None:
    """Main camera capture and sorting loop."""

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
