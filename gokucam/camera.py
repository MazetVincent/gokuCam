"""Picking the real webcam and opening the virtual one (OBS Virtual Camera)."""

from __future__ import annotations

import json
import platform
import subprocess

import cv2
import numpy as np

OWN_OUTPUT_MARKER = (255, 0, 255)  # BGR magenta test card, to recognise our own virtual camera


def is_virtual_camera(name: str) -> bool:
    lowered = name.lower()
    return "virtual" in lowered or lowered.startswith("obs")


def real_camera_order() -> list[tuple[int, str]]:
    """macOS: (OpenCV index, name) of the real cameras. OpenCV numbers cameras in the order
    of their unique ID (system_profiler lists them in another order); virtual cameras are
    left out. Empty list when the information is not available."""
    if platform.system() != "Darwin":
        return []
    try:
        result = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                                capture_output=True, text=True, timeout=15)
        cameras = json.loads(result.stdout).get("SPCameraDataType", [])
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    ordered = sorted(cameras, key=lambda camera: camera.get("spcamera_unique-id", ""))
    return [(index, camera.get("_name", "?")) for index, camera in enumerate(ordered)
            if not is_virtual_camera(camera.get("_name", ""))]


def shows_own_output(capture, virtual) -> bool:
    """Does the opened camera return what we send to the virtual camera? We send a magenta
    test card and look for it."""
    marker = np.empty((virtual.height, virtual.width, 3), np.uint8)
    marker[:] = OWN_OUTPUT_MARKER
    for _ in range(15):
        virtual.send(marker)
        ok, frame = capture.read()
        if ok:
            blue, green, red = frame.reshape(-1, 3).mean(axis=0)
            if blue > 180 and red > 180 and green < 80:
                return True
    return False


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture | None:
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def open_real_camera(index: int | None, width: int, height: int, fps: int,
                     virtual) -> tuple[cv2.VideoCapture, np.ndarray]:
    """Opens the requested webcam, or the first real one; never our own virtual camera."""
    if index is not None:
        candidates = [(index, f"index {index}")]
    else:
        candidates = real_camera_order()
        known = {i for i, _ in candidates}
        candidates += [(i, f"index {i}") for i in range(4) if i not in known]
    for i, name in candidates:
        capture = open_camera(i, width, height, fps)
        if capture is None:
            continue
        ok, frame = capture.read()
        if ok and not (virtual is not None and shows_own_output(capture, virtual)):
            print(f"[GokuCam] Camera: {name} (index {i})")
            return capture, frame
        capture.release()
    raise SystemExit(
        "No usable webcam. On macOS: System Settings > Privacy & Security > Camera, allow the app "
        "that launches GokuCam, then try again. You can also force one with --camera N."
    )


def open_virtual_camera(width: int, height: int, fps: int, backend: str | None = None):
    """pyvirtualcam camera, or None (preview only) when no driver is available."""
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(width=width, height=height, fps=fps,
                                     fmt=pyvirtualcam.PixelFormat.BGR, backend=backend)
    except Exception as error:  # pyvirtualcam raises different types depending on the OS
        print(
            "[GokuCam] No virtual camera: preview only.\n"
            "  -> Install OBS Studio, open it once, click 'Start Virtual Camera' (allow the\n"
            "     system extension), then stop it and quit OBS.\n"
            f"  Details: {error}"
        )
        return None
    print(f"[GokuCam] Virtual camera ready: {camera.device}. Select it in Teams / Slack.")
    return camera
