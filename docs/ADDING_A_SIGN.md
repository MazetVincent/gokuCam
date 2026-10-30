# Adding a sign

*Version française : [fr/AJOUTER_UN_SIGNE.md](fr/AJOUTER_UN_SIGNE.md)*

In GokuCam, a sign and the effect it triggers form a **jutsu**. Adding one means writing a small class and registering it. The application then handles everything else: hold detection and progress bar, keyboard shortcut, debug overlay, frozen image while you are invisible, and the self-test.

- [1. The contract](#1-the-contract)
- [2. Describing the sign: `checks()`](#2-describing-the-sign-checks)
- [3. Tuning thresholds on your hands](#3-tuning-thresholds-on-your-hands)
- [4. Drawing the effect: `render()`](#4-drawing-the-effect-render)
- [5. Registering the jutsu](#5-registering-the-jutsu)
- [6. Testing](#6-testing)
- [7. Complete example](#7-complete-example)
- [Checklist](#checklist)

## 1. The contract

`gokucam/jutsus/base.py`:

```python
class Jutsu(ABC):
    name: str               # identifier: "wave_goodbye"
    title: str              # preview hint: "Wave goodbye"
    short: str              # keyboard help line: "Wave"
    key: str                # shortcut that plays the effect without the sign: "w"
    hint: str               # how to make the sign: "open palm, all fingers extended"
    vanish_seconds: float
    arrival_seconds: float

    def checks(self, signs: SignInput) -> dict[str, bool]: ...           # required
    def render(self, scene: Scene, t: float, arriving: bool) -> np.ndarray: ...  # required
    def start(self, scene: Scene, arriving: bool) -> None: ...           # optional
    def hidden_image(self, scene: Scene) -> np.ndarray: ...              # optional, default: the empty room
    def leftover_box(self) -> Box | None: ...                            # optional
```

## 2. Describing the sign: `checks()`

`checks()` returns **named criteria**. The sign is made when they are all `True`. Pick readable names: `--debug` shows the missing ones live ("Wave: missing pinky extended").

### What you receive

```python
@dataclass
class SignInput:
    hands: list[np.ndarray]       # (21, 2) landmarks per tracked hand, in pixels of the 640×360 detection image
    face: Face | None             # face.forehead (x, y) and face.width; None if no face was seen recently
    still_hands: list[np.ndarray] # (lazy) hands re-detected on the still image, see below
```

y grows **downwards**: "above" means a smaller y.

### Hand landmarks

```
    4 thumb tip        8 index tip      12 middle tip      16 ring tip      20 pinky tip
    3                  7                11                 15               19
    2                  6 index PIP      10 middle PIP      14 ring PIP      18 pinky PIP
    1                  5 index MCP       9 middle MCP      13 ring MCP      17 pinky MCP
                                    0 wrist
```

The constants (`WRIST`, `INDEX_TIP`, `MIDDLE_MCP`…) are in `gokucam/hands.py`.

### Helpers (`gokucam/hands.py`)

| Helper | Meaning |
|---|---|
| `finger_extended(hand, "index")` | fingertip clearly farther from the wrist than the middle joint |
| `finger_folded(hand, "ring")` | fingertip back at the level of the middle joint |
| `index_up(hand)` | index extended, tip at least one hand-scale above the wrist |
| `hand_scale(hand)` | wrist → middle knuckle distance: **the unit for every distance on a hand** |
| `dist(hand, a, b)` | distance between two landmarks |
| `best_hand(hands, fn)` | runs `fn(hand)` on every hand and returns the checks of the hand that meets the most criteria |
| `FINGERS` | `"index"`, `"middle"`, `"ring"`, `"pinky"` |

### Rules of thumb

- **Express every distance relative to something in the image**: `hand_scale(hand)` for distances on a hand, `face.width` for distances to the face. Never use raw pixels: the sign has to work 40 cm or 1.5 m from the camera.
- **One-hand signs**: wrap the criteria in `best_hand(signs.hands, lambda hand: {...})`.
- **Two-hand signs**: look at `seal_checks()` in `jutsus/substitution.py`. It picks the pair of hands whose fingertips are closest.
- **Clasped or overlapping hands**: video tracking may merge them into one hand. Ask for `signs.still_hands` (a detection on the still image, without tracking), but **only when the merged-hand pose is seen**: it costs about 10 ms per frame.
- **Do not collide with existing signs**: add an excluding criterion. Peace out has "away from the forehead" so that it cannot fire when you do Instant Transmission; Substitution has "below the eyes".
- **Avoid everyday gestures**: a sign you make without thinking (touching your face, a thumbs up during a call) will make you vanish by accident. Substitution checks for two hands precisely so that a "shush" gesture is not enough.
- The face may be `None` (hands often hide it). Decide what that means for your sign: Instant Transmission requires a face; Substitution and Peace out treat "no face" as "far from the face".

## 3. Tuning thresholds on your hands

1. **Live**: `./run.command --debug`. Each sign gets a line in the preview with its missing criteria. Make your sign and watch which one stays missing.
2. **On a photo**: take a photo of the sign from your usual seat (a screenshot of the GokuCam preview works), then:
   ```bash
   .venv/bin/python tools/inspect_sign.py photo.jpg --out annotated.jpg --json tests/data/my_sign.json
   ```
   It prints every criterion of every jutsu, saves the photo with the landmarks drawn (video tracking in green, still-image detection in magenta), and saves the landmarks as a test fixture.
3. Check the numbers that matter (distances divided by `hand_scale`) and set the threshold **halfway** between "sign made" and the closest "sign not made" pose, so that it tolerates a sloppy sign without catching nearby gestures.

This is how the Substitution sign was calibrated: the tool showed that video tracking saw **one** merged hand while the still image saw two, which led to `single_index_up` + `still_hands`.

## 4. Drawing the effect: `render()`

```python
def render(self, scene: Scene, t: float, arriving: bool) -> np.ndarray
```

Called for every frame of the effect, with `t` going from 0 to 1 over `vanish_seconds` (departure) or `arrival_seconds` (arrival).

### What you receive

```python
@dataclass
class Scene:
    frame: np.ndarray          # live webcam frame (uint8 BGR, full resolution)
    plate: np.ndarray          # photo of the empty room
    hidden: np.ndarray         # what the call currently sees while invisible (room photo, or a log…)
    mask: np.ndarray           # person mask, float32 0..1, full resolution
    seed: int                  # fixed for the whole effect: use it for random layouts
    rng: np.random.Generator   # new values every frame: use it for flicker and jitter
    leftover: Box | None       # what the previous departure left in `hidden`
    size                       # (width, height)
    person_box                 # (x0, y0, x1, y1) bounding box of the mask
```

### The rules

- **No jump at either end.** Departure: `t = 0` must look like `scene.frame` and `t → 1` like `hidden_image(scene)` (by default `scene.plate`). Arrival: `t = 0` must look like `scene.hidden` and `t → 1` like `scene.frame`. The app switches to the frozen image (or to live video) as soon as `t` reaches 1.
- **On arrival, start from `scene.hidden`, not `scene.plate`**: you may be coming back from another jutsu's departure (for example with a log on screen).
- **Stay under about 30 ms per frame at 1280×720.** Detection already takes about 20 ms. Do heavy work at reduced resolution (see the smoke, computed at 1/4) and do per-effect work once in `start()` (see the log sprite).
- **Stable vs. lively randomness**: anything that should stay in place during the effect comes from `np.random.default_rng(scene.seed)`; flicker comes from `scene.rng`.
- **Touching only the person**: apply the same deformation to `scene.frame` and `scene.mask`, then composite over the room (see `render_teleport`). `gokucam.fx.remove_person(frame, plate, mask)` gives you the frame without the person.

### Optional hooks

- `start(scene, arriving)`: called once when the effect starts. Freeze the person (Peace out keeps a `Ghost`), generate sprites, place particles.
- `hidden_image(scene)`: what the call sees once you have vanished. Substitution returns the room with the log.
- `leftover_box()`: the region where `hidden_image()` differs from the room photo. It is passed to the next arrival as `scene.leftover` (Substitution's smoke covers the log on the way back).

Useful helpers in `gokucam/fx.py`: `smoothstep(a, b, t)` to schedule phases, `remove_person`, `pixelate`, `fractal_noise`.

## 5. Registering the jutsu

In `gokucam/jutsus/__init__.py`:

```python
from .wave_goodbye import WaveGoodbye

def default_jutsus() -> list[Jutsu]:
    return [InstantTransmission(), Substitution(), PeaceOut(), WaveGoodbye()]
```

Pick a `key` that is not already used: `b`, `d`, `s` and `q` belong to the app, and `t`, `n`, `p` to the existing jutsus.

## 6. Testing

- **The self-test plays your effect automatically** (departure, arrival, and crossings with the other jutsus), with the real models on synthetic frames:
  ```bash
  ./run.command --self-test
  ```
- **Unit tests** for the sign, with the synthetic hands of `tests/helpers.py`, or the fixture recorded with `tools/inspect_sign.py`:
  ```python
  from gokucam.jutsus.wave_goodbye import WaveGoodbye
  from tests.helpers import FACE_ABOVE, make_hand, signs

  class WaveGoodbyeTest(unittest.TestCase):
      jutsu = WaveGoodbye()

      def test_open_palm(self):
          self.assertTrue(self.jutsu.detected(signs([make_hand(open_palm=True)], FACE_ABOVE)))

      def test_two_fingers_are_not_enough(self):
          self.assertFalse(self.jutsu.detected(signs([make_hand()], FACE_ABOVE)))
  ```
- **Effect tests**: check the first and last frames (no jump), then one or two key phases. See `tests/test_effects.py`.
- `tests/test_app.py` shows how to run the whole state machine with a fake jutsu and fake detectors, without MediaPipe.

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

## 7. Complete example

`gokucam/jutsus/wave_goodbye.py`: an open palm, and you fade away. This example is checked to run end to end.

```python
"""Wave goodbye: an open palm, and you fade away."""

from __future__ import annotations

import cv2
import numpy as np

from ..fx import smoothstep
from ..hands import FINGERS, best_hand, finger_extended
from .base import Jutsu, Scene, SignInput


class WaveGoodbye(Jutsu):
    name = "wave_goodbye"
    title = "Wave goodbye"
    short = "Wave"
    key = "w"
    hint = "open palm, all fingers extended"
    vanish_seconds = 0.8
    arrival_seconds = 0.8

    def checks(self, signs: SignInput) -> dict[str, bool]:
        return best_hand(signs.hands, lambda hand: {
            f"{finger} extended": finger_extended(hand, finger) for finger in FINGERS
        })

    def render(self, scene: Scene, t: float, arriving: bool) -> np.ndarray:
        absent = scene.hidden if arriving else scene.plate
        gone = smoothstep(0.0, 1.0, 1.0 - t if arriving else t)  # 0 = you, 1 = vanished
        room = cv2.addWeighted(scene.frame, 1.0 - gone, absent, gone, 0)
        you = (scene.mask * (1.0 - gone)).astype(np.float32)
        return cv2.blendLinear(scene.frame, room, you, 1.0 - you)
```

At `gone = 0` the result is the live frame; at `gone = 1` it is exactly `absent`, so both ends join the frozen image and live video without a jump.

Beware: an open palm is a common gesture in a call (a wave, explaining something). For a real sign, add a criterion that makes it deliberate, for example "hand above the head" (`hand[WRIST][1] < face.forehead[1]`).

## Checklist

- [ ] `checks()` only uses relative thresholds (`hand_scale`, `face.width`)
- [ ] criteria names read well in `--debug`
- [ ] the sign cannot be confused with the existing ones or with an everyday gesture
- [ ] `render()` starts and ends exactly on the expected images, departure and arrival
- [ ] under about 30 ms per frame; heavy work in `start()` or at reduced resolution
- [ ] registered in `default_jutsus()` with a free `key`
- [ ] unit tests for the sign; `--self-test` passes
- [ ] both READMEs updated (sign table)
