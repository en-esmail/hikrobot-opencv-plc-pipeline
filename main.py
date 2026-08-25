import time
from ctypes import byref, memset, sizeof, cast, POINTER

import cv2

from logging_config import setup_logging
from health_checker import SystemHealthChecker
from camera_io import *          # SDK bootstrap + frame_to_bgr + MvCamera, MV_FRAME_OUT, etc.
from vision_detection import detect_objects, process_frame
from plc_controller import PLCController
from sorting import run_sorting

logger = setup_logging()

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