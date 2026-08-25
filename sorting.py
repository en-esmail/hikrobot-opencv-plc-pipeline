from typing import List, Dict, Any

from logging_config import get_logger
from constants import SortingRulesConstants, PLCConstants
from plc_controller import PLCController, PLCConnectionError, PLCTimeoutError, PLCInvalidResponseError, PLCOperationError


# ---------------------------------------------------------------------------
# 4. Which detections count as "target" vs "reject"
# ---------------------------------------------------------------------------
def is_target_object(shape: str, color: str) -> bool:
    """Determine if object should go to target or reject bin.

    Args:
        shape: "Square", "Circle", or "Triangle"
        color: "Red", "Yellow", or "Blue"

    Returns:
        True if object is target (goes to good bin), False if reject
    """
    return (shape == SortingRulesConstants.TARGET_SHAPE and
            color == SortingRulesConstants.TARGET_COLOR)


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
