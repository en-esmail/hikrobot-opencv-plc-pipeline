import socket
from typing import List, Dict, Any, Optional, Tuple

from exceptions import (
    PLCError, PLCConnectionError, PLCTimeoutError,
    PLCInvalidResponseError, PLCOperationError
)
from logging_config import get_logger
from retry_logic import retry_with_backoff
from constants import PLCConstants, RetryConstants, SortingRulesConstants
from calibration import pixel_to_robot


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
        self.z = z_above  # height above grab/place point
        self.H = z_grab  # actual grab/place point height
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

                if (shape == SortingRulesConstants.TARGET_SHAPE and
                        color == SortingRulesConstants.TARGET_COLOR):
                    self.logger.debug(f"Object is target, placing in target bin")
                    self._place_target()
                else:
                    self.logger.debug(f"Object is reject, placing in reject bin")
                    self._place_reject()

                self.logger.info(f"Successfully processed object {i + 1}/{len(detections)}")
            except (PLCError, Exception) as e:
                self.logger.error(f"Failed to process detection {i + 1}/{len(detections)}: {e}", exc_info=True)
                raise
