import os
import sys
from ctypes import byref, memset, sizeof, cast, POINTER, c_ubyte

import numpy as np

from exceptions import FrameConversionError
from logging_config import get_logger

logger = get_logger()

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
