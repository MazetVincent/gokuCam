# GokuCam

**A virtual webcam for Teams, Slack, Zoom or Meet: make a hand sign and vanish from the call.**

GokuCam sits between your real webcam and your video-call app. It watches your hands; when you make one of the signs below, it plays an anime-style vanishing effect, then shows the call a frozen photo of your empty room until you make a sign again.

*Lire en français : [README.fr.md](README.fr.md)*

| Sign | Effect |
|---|---|
| **Instant Transmission** (Goku): index and middle finger together on your forehead | Blue energy lines appear around you, strips of your body tear vertically, then your whole body gathers into a thin column and disappears |
| **Substitution** (Naruto): hands clasped with fingers interlocked, both index fingers up and touching, in front of your chin | You vanish in a cloud of smoke; when it clears, a tree log stands where you were sitting |
| **Peace out**: a V sign, away from your face | You turn into a translucent ghost of compression blocks, a few blocks smear sideways, and you are gone (the "bye" meme) |

Make any sign again to come back; the return plays that sign's arrival effect.

Everything runs locally. No image is recorded or sent anywhere except to the call itself. The only network access is the first-run download of three official MediaPipe models from Google.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How it works (overview)](#how-it-works-overview)
- [Project layout](#project-layout)
- [Adding your own sign](#adding-your-own-sign)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

## Requirements

- **macOS on Apple Silicon** (developed and tested on an M1 MacBook). The code also targets Windows and Linux (MediaPipe and pyvirtualcam support them), but it has not been tested there.
- **Python 3.10 to 3.14**.
- **[OBS Studio](https://obsproject.com/) 30+**, only for its virtual camera driver. On Linux, [v4l2loopback](https://github.com/umlaeute/v4l2loopback) works instead.
- A webcam.

## Installation

1. **Install OBS Studio** and enable its virtual camera once:
   ```bash
   brew install --cask obs
   ```
   Open OBS and click **Start Virtual Camera**. macOS asks you to allow a system extension in *System Settings > Privacy & Security*. Then click **Stop Virtual Camera** and quit OBS. You will not need OBS again: GokuCam drives the virtual camera itself.

2. **Get the code** and launch it:
   ```bash
   git clone https://github.com/<you>/gokucam.git
   cd gokucam
   ./run.command
   ```
   The first launch creates a virtual environment in `.venv/`, installs the dependencies and downloads the models into `models/` (about 8 MB). On macOS you can also double-click `run.command` in Finder.

3. **Allow camera access** when macOS asks. The permission goes to the app that launches GokuCam: Terminal, iTerm, your IDE…

## Usage

1. **Step out of the frame** during the 5-second countdown: GokuCam takes a photo of your empty room. The photo is then shown as a thumbnail for 4 seconds; if you are in it, press `B` to retake it.
2. In your video-call app, pick **OBS Virtual Camera** as the camera:
   - **Teams**: *Settings > Devices > Camera*
   - **Slack** (huddles): the arrow next to the camera button, or *Preferences > Audio & video*
   - Quit and reopen the app (Cmd+Q) if the camera does not appear.
3. **Hold a sign for about 0.3 s** to vanish, and again to come back.

Keep the preview window open during the call: closing it stops the virtual camera. Hints and debug info only appear in the preview; the call always gets a clean image.

| Key (in the preview) | Action |
|---|---|
| `T` / `N` / `P` | Play Instant Transmission / Substitution / Peace out without the sign |
| `B` | Retake the room photo |
| `D` | Debug: for each sign, show which criteria are missing |
| `S` | Save a diagnostic snapshot (camera, output, room photo, mask) to `captures/` |
| `Q` / `Esc` | Quit |

Command-line options (`./run.command --help`):

| Option | Meaning |
|---|---|
| `--debug` | Start with the debug overlay on |
| `--camera N` | Force a webcam index (by default the first real camera; virtual cameras are skipped) |
| `--width`, `--height`, `--fps` | Capture and output format (default 1280×720 at 30 fps) |
| `--no-virtual` | Preview only, no virtual camera |
| `--backend` | pyvirtualcam backend: `obs`, `unitycapture` or `v4l2loopback` |
| `--no-mirror` | Do not mirror the preview |
| `--self-test` | Load the models and play every effect on synthetic frames, without a webcam |

**Tips**
- The room photo is frozen. If the light changes (lamp switched on, nightfall), coming back will show it: press `B` to retake the photo.
- Do not move the camera or your chair after the photo.
- Light coming from behind your screen (facing you) gives the best results.

## How it works (overview)

```
webcam ──> detection (640×360) ──> signs ──> state machine ──> effect ──> virtual camera ──> Teams / Slack
            · person mask                     plate → visible → vanishing → hidden → appearing → visible
            · hand landmarks
            · face
```

- **Detection**: three MediaPipe models run on a downscaled copy of every frame: a selfie segmenter (person mask), a hand landmarker (21 points per hand) and a face detector (to locate the forehead).
- **Signs**: each sign is a set of named geometric criteria on the landmarks, for example "index extended" or "fingers together". Thresholds are relative to the size of the hand, so signs work near and far from the camera. A sign must be held for 0.35 s and fires once, until it is released.
- **Invisibility**: rather than erasing you from the live image, which leaves shadows, halos and exposure flicker, GokuCam sends the call a **photo of the empty room** taken at startup, with a little animated sensor grain. Live video only comes back with the next sign.
- **Effects**: each effect is a function of `t` from 0 to 1 that composites the live frame, the person mask and the room photo with OpenCV and NumPy. They run in real time: 13 to 28 ms per frame on an M1.

The full story, including the design decisions and what did not work, is in **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)**.

## Project layout

```
gokucam/
  cli.py            command line, main loop, self-test
  app.py            state machine: room photo, signs, effects, frozen image
  models.py         MediaPipe model download and detectors
  hands.py          hand landmark indices, finger geometry, gesture tracker
  camera.py         picks the real webcam, opens the virtual camera
  preview.py        preview window overlay, diagnostic snapshots
  fx.py             small image helpers (smoothstep, noise, pixelate…)
  jutsus/
    base.py         the Jutsu contract: checks() + render()
    instant_transmission.py
    substitution.py
    peace_out.py
    __init__.py     registry of jutsus
tests/              unit tests (no webcam or MediaPipe needed)
tools/              inspect_sign.py: what the detectors see of a sign on a photo
docs/               how it works, adding a sign (English and French)
run.command         launcher (creates .venv on first run)
```

## Adding your own sign

A sign and its effect form a **jutsu**: a small class with two methods.

```python
class Wave(Jutsu):
    name, title, short, key, hint = "wave", "Wave goodbye", "Wave", "w", "open palm, fingers spread"
    vanish_seconds = arrival_seconds = 0.8

    def checks(self, signs):                 # named criteria; the sign is made when all are True
        return best_hand(signs.hands, lambda hand: {
            "all fingers extended": all(finger_extended(hand, f) for f in FINGERS),
        })

    def render(self, scene, t, arriving):    # frame t of the effect, t from 0 to 1
        ...
```

Add it to `default_jutsus()` in `gokucam/jutsus/__init__.py`, and the app does the rest: hold detection, keyboard shortcut, debug overlay, frozen image, self-test. The step-by-step guide, with the landmark map, the geometry helpers, how to tune thresholds on your own hands and how to test, is in **[docs/ADDING_A_SIGN.md](docs/ADDING_A_SIGN.md)**.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -t .   # 42 tests, a few seconds, no webcam needed
./run.command --self-test                             # full pipeline with the real models
```

The tests cover the signs (including landmarks recorded from a real photo of the Substitution sign), every effect's start and end frames and key phases, the state machine with a fake jutsu, the room photo and the camera selection.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No usable webcam" | Allow camera access for the app that launches GokuCam (*System Settings > Privacy & Security > Camera*), then relaunch |
| "No virtual camera: preview only" | Enable the OBS virtual camera once (see Installation), then quit OBS |
| The camera does not show up in Teams / Slack | Quit the app completely (Cmd+Q) and reopen it while GokuCam is running |
| A brief magenta flash in the call at startup | Normal: GokuCam checks that it is not reading its own output |
| A sign does not trigger | Press `D` and read which criterion is missing; see [docs/ADDING_A_SIGN.md](docs/ADDING_A_SIGN.md#3-tuning-thresholds-on-your-hands) |
| You can see yourself in the room photo | Press `B` and step fully out of the frame during the countdown |

## Credits

- [MediaPipe](https://ai.google.dev/edge/mediapipe) (Google) for segmentation, hand landmarks and face detection
- [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/) for all the image processing
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) and [OBS Studio](https://obsproject.com/) for the virtual camera

The effects are inspired by *Dragon Ball Z*, *Naruto* and an internet meme. This is a fan project, not affiliated with their authors or publishers; no image from those works is included: everything is drawn in code.

## License

[MIT](LICENSE)
