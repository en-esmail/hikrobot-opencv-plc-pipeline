"""
Centralized constants for the hikrobot-opencv-plc-pipeline project.

Consolidates all magic numbers and configuration values into a single,
easy-to-maintain module. Updated here, changes propagate everywhere.
"""

# =============================================================================
# IMAGE PROCESSING CONSTANTS
# =============================================================================

class ImageProcessingConstants:
    """Parameters for frame preprocessing and object detection."""
    
    # Gaussian blur
    GAUSSIAN_BLUR_KERNEL = (5, 5)
    GAUSSIAN_BLUR_SIGMA = 0.0
    
    # Morphological operations
    MORPH_KERNEL = (5, 5)
    
    # Thresholding
    THRESHOLD_VALUE = 127
    THRESHOLD_MAX = 255
    
    # Contour filtering
    MIN_CONTOUR_AREA_PX2 = 500
    
    # Shape classification
    SQUARE_ASPECT_RATIO_MIN = 0.80
    SQUARE_ASPECT_RATIO_MAX = 1.20
    CIRCLE_CIRCULARITY_MIN = 0.75
    POLYGON_SOLIDITY_MIN = 0.90
    CIRCLE_VERTEX_COUNT_MIN = 7
    
    # Drawing parameters
    CONTOUR_LINE_WIDTH = 2
    LABEL_BACKGROUND_COLOR = (255, 255, 255)
    LABEL_TEXT_COLOR = (0, 0, 255)  # Red
    LABEL_CHAR_WIDTH = 12
    LABEL_Y_OFFSET = 25
    DRAW_COLOR = (0, 255, 0)
    
    # Pixel approximation
    CONTOUR_APPROX_EPSILON = 0.04


# =============================================================================
# COLOR DETECTION CONSTANTS
# =============================================================================

class ColorConstants:
    """HSV color ranges for object detection."""
    
    # HSV ranges: (lower_H, lower_S, lower_V, upper_H, upper_S, upper_V)
    # Red wraps around hue circle, needs two ranges
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
    
    # Valid colors for detection
    VALID_COLORS = ["Red", "Yellow", "Blue"]
    
    # Default color for unknown objects
    DEFAULT_COLOR = "Unknown"
    MIN_COLOR_MATCH_RATIO = 0.10


# =============================================================================
# PLC COMMUNICATION CONSTANTS
# =============================================================================

class PLCConstants:
    """Parameters for PLC communication and robot control."""
    
    # Connection parameters
    DEFAULT_IP = "192.168.6.10"
    DEFAULT_PORT = 2023
    SOCKET_BUFFER_SIZE = 2048
    SOCKET_TIMEOUT_SECONDS = 5
    
    # Motion parameters
    DEFAULT_Z_HOVER = -30  # mm above target
    DEFAULT_Z_GRASP = -84  # mm at grasp point
    DEFAULT_ANGLE = 0      # degrees
    
    # Response validation
    MAX_RESPONSE_LENGTH = 100  # max expected response chars
    EXPECTED_RESPONSE_PATTERN = r"^\d+$"  # numeric only
    
    # Response codes
    PLC_READY_RESPONSE = "1"
    PLC_NOT_READY_RESPONSE = "0"
    
    # Command encoding
    COMMAND_ENCODING = "utf-8"
    RESPONSE_ENCODING = "gbk"
    
    # PLC command codes
    HANDSHAKE_CMD = "plc,6"
    MOVE_CMD_TEMPLATE = "plc,{x},{y},{z},{angle},7"
    SUCTION_OPEN_CMD = "plc,4"  # grab/open suction
    SUCTION_CLOSE_CMD = "plc,5"  # release/close suction
    
    # Grid positioning (for placement)
    GRID_SLOTS_PER_LAYER = 6  # 2 rows x 3 columns
    GRID_ROWS = 2
    GRID_COLUMNS = 3
    GRID_CELL_SIZE = 45  # mm between slots
    
    # Target bin positioning
    TARGET_BIN_ORIGIN_X = -101  # mm
    TARGET_BIN_ORIGIN_Y = 55.5  # mm
    
    # Reject bin positioning
    REJECT_BIN_ORIGIN_X = 160.5  # mm
    REJECT_BIN_ORIGIN_Y = 60.5   # mm


# =============================================================================
# CALIBRATION CONSTANTS
# =============================================================================

class CalibrationConstants:
    """Camera to robot coordinate transformation parameters."""
    
    # Default placeholder calibration matrix (identity-like transform)
    # Replace with actual calibration after measuring correspondences
    # Maps pixel coordinates (px, py) to robot coordinates (X, Y) in mm
    PIXEL_TO_ROBOT_MATRIX = [
        [-0.001169, -0.086443, 94.633716],
        [-0.085307, 0.001197, 89.196052],
    ]
    
    # Calibration points (for reference, update after re-calibration)
    # Format: (pixel_x, pixel_y, robot_x, robot_y)
    # Collect 3-4 points by jogging robot to known positions
    CALIBRATION_POINTS = [
        # (pixel_x, pixel_y, robot_x, robot_y),  # Point 1
        # (pixel_x, pixel_y, robot_x, robot_y),  # Point 2
        # (pixel_x, pixel_y, robot_x, robot_y),  # Point 3
    ]


# =============================================================================
# CAMERA PARAMETERS
# =============================================================================

class CameraConstants:
    """Hikrobot camera configuration parameters."""
    
    # Trigger mode
    TRIGGER_MODE = 0  # Off (continuous acquisition)
    
    # Exposure settings
    EXPOSURE_AUTO = 0  # Manual exposure
    EXPOSURE_TIME = 10000.0  # microseconds
    
    # Gain settings
    GAIN = 5.0  # dB
    
    # Frame buffer
    FRAME_GRAB_TIMEOUT = 1000  # milliseconds
    FRAME_GRAB_SLEEP = 0.005  # seconds (if frame not ready)
    
    # Camera window
    DISPLAY_WINDOW_WIDTH = 800
    DISPLAY_WINDOW_HEIGHT = 600
    DISPLAY_WINDOW_NAME = "Hikrobot Camera"


# =============================================================================
# HEALTH CHECK CONSTANTS
# =============================================================================

class HealthCheckConstants:
    """Parameters for system health validation."""
    
    # Connection checks
    CONNECTIVITY_CHECK_TIMEOUT = 3  # seconds
    
    # Handshake check
    HANDSHAKE_CHECK_COMMAND = "plc,6"
    HANDSHAKE_EXPECTED_RESPONSE = "1"
    
    # Health check report formatting
    REPORT_WIDTH = 60
    REPORT_SEPARATOR = "=" * REPORT_WIDTH


# =============================================================================
# LOGGING CONSTANTS
# =============================================================================

class LoggingConstants:
    """Logging configuration parameters."""
    
    # Directories
    LOG_DIRECTORY = "logs"
    
    # File names
    DEBUG_LOG_FILE = "debug.log"
    INFO_LOG_FILE = "info.log"
    ERROR_LOG_FILE = "errors.log"
    
    # Rotation parameters
    MAX_LOG_FILE_SIZE = 10_000_000  # 10 MB
    BACKUP_COUNT_DEBUG = 5
    BACKUP_COUNT_ERROR = 3
    
    # Log levels
    DEFAULT_LOG_LEVEL = "INFO"
    
    # Date format
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Formatters
    DETAILED_FORMAT = (
        "%(asctime)s | %(name)s | %(levelname)-8s | "
        "%(funcName)s:%(lineno)d | %(message)s"
    )
    SIMPLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
    CONSOLE_FORMAT = "%(levelname)-8s | %(message)s"


# =============================================================================
# SORTING RULES CONSTANTS
# =============================================================================

class SortingRulesConstants:
    """Object sorting classification rules."""
    
    # Target object definition (goes to "good" bin)
    TARGET_SHAPE = "Square"
    TARGET_COLOR = "Yellow"
    
    # Valid shapes for detection
    VALID_SHAPES = ["Square", "Circle", "Triangle"]
    
    # Placement strategy
    PLACE_TARGET_BIN = "target"
    PLACE_REJECT_BIN = "reject"


# =============================================================================
# RETRY LOGIC CONSTANTS
# =============================================================================

class RetryConstants:
    """Parameters for operation retry logic."""
    
    # Retry configuration defaults
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_INITIAL_DELAY = 0.1  # seconds
    DEFAULT_BACKOFF_FACTOR = 2.0
    DEFAULT_MAX_DELAY = 30.0  # seconds (cap on backoff)
    
    # Exceptions that should trigger retry
    RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError)


# =============================================================================
# CONVENIENCE HELPER: Get all constants as a dictionary
# =============================================================================

def get_all_constants() -> dict:
    """Return all constants as a dictionary for reference."""
    return {
        "ImageProcessing": ImageProcessingConstants.__dict__,
        "Colors": ColorConstants.__dict__,
        "PLC": PLCConstants.__dict__,
        "Calibration": CalibrationConstants.__dict__,
        "Camera": CameraConstants.__dict__,
        "HealthCheck": HealthCheckConstants.__dict__,
        "Logging": LoggingConstants.__dict__,
        "SortingRules": SortingRulesConstants.__dict__,
        "Retry": RetryConstants.__dict__,
    }


if __name__ == "__main__":
    # Print all constants for reference
    import json
    print(json.dumps(get_all_constants(), indent=2, default=str))
