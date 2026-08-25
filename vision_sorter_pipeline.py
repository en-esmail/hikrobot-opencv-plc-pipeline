"""
Backward-compatible entry point for the split vision sorter pipeline.

The implementation now lives in focused modules:
- `camera_io.py` for Hikrobot SDK setup and frame conversion
- `vision_detection.py` for shape and color detection
- `calibration.py` for pixel-to-robot coordinate conversion
- `plc_controller.py` for PLC communication and robot motion
- `sorting.py` for target/reject sorting orchestration
- `main.py` for the live camera loop

Run this file only if older tooling still calls `vision_sorter_pipeline.py`.
New code should import from the focused modules directly.
"""

from calibration import pixel_to_robot
from plc_controller import PLCController
from sorting import is_target_object, run_sorting
from vision_detection import (
    ObjectDetector,
    classify_color,
    classify_shape,
    detect_objects,
    get_detector,
    process_frame,
)


def main() -> None:
    """Run the split pipeline's main capture loop."""
    from main import main as run_main

    run_main()


if __name__ == "__main__":
    main()
