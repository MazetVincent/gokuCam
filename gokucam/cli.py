"""Command line: `python -m gokucam [options]`."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from .app import GokuCam
from .camera import open_real_camera, open_virtual_camera
from .models import Detectors, ensure_models
from .preview import PREVIEW_WINDOW, build_preview, save_snapshot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gokucam", description="Virtual webcam for Teams / Slack: make a hand sign and vanish.")
    parser.add_argument("--camera", type=int, help="webcam index (default: the first real camera)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-virtual", action="store_true", help="preview only, no virtual camera")
    parser.add_argument("--backend", help="pyvirtualcam backend (obs, unitycapture, v4l2loopback)")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the preview")
    parser.add_argument("--debug", action="store_true", help="show which criteria of each sign are met")
    parser.add_argument("--self-test", action="store_true",
                        help="run the whole pipeline on synthetic frames, without a webcam")
    return parser.parse_args(argv)


def self_test() -> int:
    """Loads the models and plays every jutsu (departure and arrival, including arriving
    with a different jutsu than the one used to vanish) and every overlay on synthetic frames."""
    size = (1280, 720)
    rng = np.random.default_rng(0)
    with Detectors() as detectors:
        app = GokuCam(detectors, size)
        names = list(app.jutsus)
        steps = [(None, "visible")]
        for name in names:
            steps += [(name, "hidden"), (name, "visible")]
        for leave, back in zip(names, names[1:] + names[:1]):
            steps += [(leave, "hidden"), (back, "visible")]
        now = 0.0
        for name, expected in steps:
            if name:
                app.trigger(now, name)
            for _ in range(70):  # 7 s in steps of 0.1 s: longer than any effect
                frame = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
                out = app.step(frame, now)
                assert out.shape == frame.shape and out.dtype == np.uint8
                if app.state == "hidden":  # frozen image, independent of the live feed
                    assert np.abs(out.astype(np.int16) - app.hidden_image).mean() < 3
                build_preview(app, out, now, mirror=True, debug=True)
                now += 0.1
            assert app.state == expected, (name, app.state, expected)
            print(f"[self-test] {name or 'room photo'} -> {app.state} OK")
        for name in app.overlays:  # switched on by hand, then off: fades in, draws, fades out
            app.toggle_overlay(name, now)
            for _ in range(40):
                frame = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
                out = app.step(frame, now)
                now += 0.1
            assert app.overlay_states[name].level == 1.0 and not np.array_equal(out, frame), name
            app.toggle_overlay(name, now)
            for _ in range(30):
                frame = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
                out = app.step(frame, now)
                now += 0.1
            assert app.overlay_states[name].level == 0.0 and np.array_equal(out, frame), name
            print(f"[self-test] overlay {name} OK")
    print("[self-test] OK: models loaded, every jutsu and overlay played.")
    return 0


def run(args: argparse.Namespace) -> int:
    virtual = None if args.no_virtual else open_virtual_camera(args.width, args.height, args.fps, args.backend)
    try:
        capture, frame = open_real_camera(args.camera, args.width, args.height, args.fps, virtual)
    except SystemExit:
        if virtual is not None:
            virtual.close()
        raise
    height, width = frame.shape[:2]
    debug = args.debug
    print("[GokuCam] Step out of the frame: the room photo is taken in a few seconds.")

    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW_WINDOW, 960, int(960 * height / width))
    try:
        with Detectors() as detectors:
            app = GokuCam(detectors, (width, height))
            jutsu_keys = {ord(jutsu.key): name for name, jutsu in app.jutsus.items()}
            overlay_keys = {ord(overlay.key): name for name, overlay in app.overlays.items()}
            app.recapture(time.monotonic())
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("[GokuCam] Webcam feed interrupted.")
                    break
                now = time.monotonic()
                output = app.step(frame, now)
                if virtual is not None:
                    if output.shape[:2] != (virtual.height, virtual.width):
                        output = cv2.resize(output, (virtual.width, virtual.height))
                    virtual.send(output)
                cv2.imshow(PREVIEW_WINDOW, build_preview(app, output, now, not args.no_mirror, debug))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in jutsu_keys:
                    app.trigger(now, jutsu_keys[key])
                elif key in overlay_keys:
                    app.toggle_overlay(overlay_keys[key], now)
                elif key == ord("b"):
                    app.recapture(now)
                elif key == ord("d"):
                    debug = not debug
                elif key == ord("s"):
                    print(f"[GokuCam] Snapshot saved: {save_snapshot(app, frame, output)}")
    finally:
        capture.release()
        if virtual is not None:
            virtual.close()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_models()
    return self_test() if args.self_test else run(args)
