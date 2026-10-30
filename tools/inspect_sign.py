"""Shows what the detectors see on a photo of a sign, and which criteria of each jutsu pass.

    .venv/bin/python tools/inspect_sign.py photo.jpg [--out annotated.jpg] [--json fixture.json]

- prints the criteria of every registered jutsu (the same ones `--debug` shows live);
- `--out` saves the photo with the landmarks drawn (video tracking in green, still-image
  detection in magenta);
- `--json` saves the landmarks, ready to use as a test fixture (see tests/data/).

Take the photo with the webcam you use for calls, from your usual seat (a screenshot of
the GokuCam preview works too): thresholds are relative, but the pose matters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gokucam.hands import face_from_detection  # noqa: E402
from gokucam.jutsus import SignInput, default_jutsus  # noqa: E402
from gokucam.models import DETECT_SIZE, Detectors, ensure_models  # noqa: E402


def to_detection_image(image: np.ndarray) -> np.ndarray:
    """Pads the photo to 16:9 (without distorting it) and resizes it like a webcam frame."""
    h, w = image.shape[:2]
    target_w = max(w, int(h * 16 / 9))
    target_h = max(h, int(target_w * 9 / 16))
    canvas = np.full((target_h, target_w, 3), 128, np.uint8)
    y0, x0 = (target_h - h) // 2, (target_w - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = image
    return cv2.resize(canvas, DETECT_SIZE, interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photo")
    parser.add_argument("--out", help="annotated image to write")
    parser.add_argument("--json", help="landmarks to write as a test fixture")
    args = parser.parse_args()

    image = cv2.imread(args.photo)
    if image is None:
        raise SystemExit(f"Cannot read {args.photo}")
    ensure_models()
    small = to_detection_image(image)
    with Detectors() as detectors:
        for _ in range(10):  # video mode: let tracking settle on the still photo
            _, hands, detection = detectors.process(small)
        still = detectors.hands_on_still_image()
    face = face_from_detection(detection, DETECT_SIZE, 0.0) if detection is not None else None

    print(f"face: {'yes' if face else 'no'}   hands (video tracking): {len(hands)}   hands (still image): {len(still)}")
    signs = SignInput(hands, face, lambda: still)
    for jutsu in default_jutsus():
        checks = jutsu.checks(signs)
        verdict = "DETECTED" if all(checks.values()) else "no"
        details = ", ".join(f"{name} {'OK' if ok else '--'}" for name, ok in checks.items())
        print(f"  {jutsu.title:32} {verdict:9} {details}")

    if args.out:
        out = small.copy()
        for color, group in (((0, 255, 0), hands), ((255, 0, 255), still)):
            for hand in group:
                for x, y in hand:
                    cv2.circle(out, (int(x), int(y)), 2, color, -1)
        if face is not None:
            cv2.circle(out, tuple(int(v) for v in face.forehead), int(0.55 * face.width), (0, 200, 255), 1)
        cv2.imwrite(args.out, cv2.resize(out, None, fx=2, fy=2))
        print(f"annotated image: {args.out}")
    if args.json:
        fixture = {"video_hands": [h.round(1).tolist() for h in hands],
                   "still_hands": [h.round(1).tolist() for h in still],
                   "face": None if face is None else {"forehead": face.forehead.round(1).tolist(),
                                                       "width": round(face.width, 1)}}
        Path(args.json).write_text(json.dumps(fixture))
        print(f"landmarks: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
