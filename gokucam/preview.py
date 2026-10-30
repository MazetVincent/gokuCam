"""Preview window: the user sees hints and debug info; the call only gets the clean image."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .app import GokuCam
from .models import ROOT

PREVIEW_WINDOW = "GokuCam - preview (Q to quit)"
CAPTURES_DIR = ROOT / "captures"


def put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.7) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def key_help(app: GokuCam) -> str:
    items = (*app.jutsus.values(), *app.overlays.values())
    signs = "  ".join(f"{item.key.upper()}: {item.short}" for item in items)
    return f"{signs}  B: retake photo  D: debug  S: snapshot  Q: quit"


def build_preview(app: GokuCam, output: np.ndarray, now: float, mirror: bool, debug: bool) -> np.ndarray:
    preview = cv2.flip(output, 1) if mirror else output.copy()
    h, w = preview.shape[:2]
    if debug:
        for (x, y), radius, color in app.debug_points():
            cv2.circle(preview, (w - 1 - x if mirror else x, y), radius, color, 2, cv2.LINE_AA)
    if app.plate is not None and now < app.review_until:  # thumbnail of the room photo to check
        thumb = cv2.resize(cv2.flip(app.plate, 1) if mirror else app.plate, (w // 3, h // 3))
        cv2.rectangle(preview, (w - w // 3 - 24, 16), (w - 16, 24 + h // 3), (255, 255, 255), -1)
        preview[20:20 + h // 3, w - w // 3 - 20:w - 20] = thumb
    for i, line in enumerate(app.hud_lines(now) + (app.debug_lines() if debug else [])):
        put_text(preview, line, (20, 40 + 32 * i))
    progress = app.gesture_progress(now)
    if progress > 0:
        cv2.rectangle(preview, (20, h - 30), (20 + int((w - 40) * progress), h - 18), (255, 220, 120), -1)
    put_text(preview, key_help(app), (20, h - 45), 0.55)
    return preview


def save_snapshot(app: GokuCam, frame: np.ndarray, output: np.ndarray) -> Path:
    """Saves, locally only, what is needed to diagnose a compositing issue."""
    folder = CAPTURES_DIR / time.strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "1_camera.jpg"), frame)
    cv2.imwrite(str(folder / "2_output.jpg"), output)
    if app.plate is not None:
        cv2.imwrite(str(folder / "3_room_photo.jpg"), app.plate)
    cv2.imwrite(str(folder / "4_mask.png"), (app.full_mask() * 255).astype(np.uint8))
    return folder
