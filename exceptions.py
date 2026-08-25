"""
Custom exception hierarchy for the hikrobot-opencv-plc-pipeline project.

Provides specific exception types for different error scenarios, improving
error handling precision and making it easier to distinguish failure modes.
"""


class HikrobotError(Exception):
    """Base exception for all hikrobot pipeline errors."""
    pass


# --- Configuration Errors ---
class ConfigurationError(HikrobotError):
    """Raised when configuration is invalid or missing."""
    pass


# --- Calibration Errors ---
class CalibrationError(HikrobotError):
    """Raised when calibration-related operations fail."""
    pass


# --- Detection Errors ---
class DetectionError(HikrobotError):
    """Raised when object detection fails."""
    pass


# --- PLC / Communication Errors ---
class PLCError(HikrobotError):
    """Base class for PLC communication and operation errors."""
    pass


class PLCConnectionError(PLCError):
    """Raised when unable to connect to PLC or connection is lost."""
    pass


class PLCTimeoutError(PLCError):
    """Raised when PLC communication times out."""
    pass


class PLCInvalidResponseError(PLCError):
    """Raised when PLC returns an invalid or malformed response."""
    pass


class PLCOperationError(PLCError):
    """Raised when a PLC operation fails (grab, place, etc.)."""
    pass


# --- Sorting Rule Errors ---
class SortingRuleError(HikrobotError):
    """Raised when sorting rule evaluation fails."""
    pass


# --- Camera / Vision Errors ---
class CameraError(HikrobotError):
    """Raised when camera operations fail."""
    pass


class FrameConversionError(CameraError):
    """Raised when frame format conversion fails."""
    pass
