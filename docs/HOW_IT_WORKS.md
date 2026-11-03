# How GokuCam works

*Version française : [fr/FONCTIONNEMENT.md](fr/FONCTIONNEMENT.md)*

This document follows one webcam frame through the program, then describes each effect and the design decisions behind them, including the approaches that were tried and dropped.

- [1. The frame loop](#1-the-frame-loop)
- [2. Detection](#2-detection)
- [3. Recognising signs](#3-recognising-signs)
- [4. The room photo and invisibility](#4-the-room-photo-and-invisibility)
- [5. The state machine](#5-the-state-machine)
- [6. The effects](#6-the-effects)
- [7. Webcam and virtual camera](#7-webcam-and-virtual-camera)
- [8. Performance](#8-performance)
- [9. Privacy](#9-privacy)

## 1. The frame loop

`cli.run()` reads the webcam at 1280×720, 30 fps, and for every frame:

1. `GokuCam.step(frame)` (in `app.py`) runs detection, updates every sign, advances the state machine and returns **the frame for the call**.
2. That frame goes to the virtual camera (`pyvirtualcam` → OBS Virtual Camera → Teams / Slack).
3. `preview.build_preview()` draws a mirrored copy with hints, the hold progress bar and optional debug info in the local preview window. **The overlay never reaches the call.**
4. Keyboard shortcuts are handled (`T`/`N`/`P`, `B`, `D`, `S`, `Q`).

## 2. Detection

`models.Detectors` wraps four official MediaPipe models. They are downloaded once into `models/`:

| Model | Output | Used for |
|---|---|---|
| `selfie_segmenter_landscape` | person probability per pixel | the person mask: where the effects apply |
| `hand_landmarker` | 21 points per hand, up to 2 hands | signs |
| `blaze_face_short_range` | face box and 6 keypoints | locating the forehead |
| `efficientdet_lite0` | object boxes (the 80 COCO categories; only "cup" is kept) | spotting a mug for "This is fine" |

They run in **video mode** on a **640×360** copy of the frame: tracking makes landmarks stable and cheap from one frame to the next, and the small size keeps detection at about 20 ms per frame on an M1.

**Person mask** (`app.refine_mask`): the raw segmentation is thresholded generously, dilated by 15 px, blurred, and kept from fading too fast (`max(mask, 0.75 × previous mask)`). Effects prefer to touch a little of the room rather than miss a piece of you.

**Forehead** (`hands.face_from_detection`): BlazeFace gives the eyes and mouth. The forehead is extrapolated above the eyes along the mouth→eyes axis, at 0.8 eye distances, so it follows head tilt. Hands often hide the face, so the last face seen is kept for one second.

**Still-image hand detection**: video tracking has a blind spot. When the hands are clasped with interlocked fingers (the Substitution sign), it tracks **a single merged hand** and never looks for a second one, whatever the confidence threshold. A second hand landmarker in **image mode** (no tracking, confidence 0.3) does find both hands on the same frame. It is only run on demand (see [3.3](#33-a-two-hand-sign-substitution)) and at most every third frame, because it costs about 35 ms.

**Object detection** costs as much as all the other models together (about 18 ms), so it runs on a **background thread**, on another CPU core, and only while a hand is in view (no hand, no mug being held). The main loop never waits for it: it uses the latest result, a few frames old at most, which is plenty for a mug.

## 3. Recognising signs

### 3.1 Criteria on landmarks

A sign is a dictionary of **named boolean criteria** computed from the landmarks (`Jutsu.checks()`). The sign is made when they are all true. The names are what `--debug` shows: "Naruto: missing both index up, index tips touching".

The building blocks are in `hands.py`:

- `hand_scale(hand)`: wrist → middle-finger knuckle distance. It barely changes with the pose, so **every distance threshold is a multiple of it**, and a sign works whether your hand is 40 cm or 1.5 m from the camera.
- `finger_extended(hand, finger)`: the fingertip is clearly farther from the wrist than the finger's middle joint (×1.1).
- `finger_folded(hand, finger)`: the fingertip comes back to the level of the middle joint (×1.05).
- `index_up(hand)`: index extended and its tip at least one hand-scale above the wrist.
- Distances to the face are multiples of the face width.

For example, Instant Transmission is index and middle extended, ring and pinky folded, fingertips closer than 0.5 hand-scale, and fingertips within 0.55 face-widths of the forehead. Peace out is the same hand but with the fingertips **more** than 0.55 hand-scale apart, and more than 0.8 face-widths from the forehead. The two signs cannot be confused.

### 3.2 Holding a sign

`hands.GestureTracker` turns "the sign is visible on this frame" into "fire now":

- the sign must be held for **0.35 s** (the progress bar in the preview);
- detection gaps up to 0.15 s do not reset the hold (landmarks flicker);
- after firing, the sign must be **released for 0.5 s** before it can fire again, so holding it through the whole effect does not bring you straight back.

### 3.3 A two-hand sign: Substitution

The Substitution sign was calibrated on a real photo (its landmarks are stored in `tests/data/joined_hands.json`):

1. Video tracking sees one hand, index up, other fingers folded (`single_index_up`).
2. Only in that case, `SignInput.still_hands` runs the still-image detection, which finds the two hands.
3. The criteria: two hands, both index fingers up, middle and ring folded on at least one hand (the rear hand is partly hidden, so its predicted fingers are unreliable), index tips closer than 0.6 hand-scale, and below the eyes.

A "shush" gesture (one finger on the lips) looks like step 1, but the still-image detection only finds one hand, so it never fires.

## 4. The room photo and invisibility

### 4.1 What the call sees while you are invisible

The call receives a **frozen image**: the photo of the empty room, or what the departure left behind (the log, for Substitution). Six variants with a little Gaussian sensor grain (σ = 1.5) are cycled, so the image looks like a live feed rather than a still. The live webcam is only used to watch for the next sign.

This is the most important design decision of the project, and it came from failing at the alternative first. Erasing the person from the live image (live background replacement) was tried for a long time:

1. **Learned background, pixel by pixel**: the background is averaged only where nobody stands. Areas never seen (behind a gaming chair you sit in all day) had to be invented by inpainting, which left a blurry, skin-coloured blob.
2. **Cast shadows**: the segmenter detects people, not their shadows. A lamp threw a shadow on the door that stayed visible when "invisible", and moving shadows got averaged into the learned background. A shadow detector was added (darker than the background, same hue, measured on real captures: brightness ratio ≈ 0.66, hue difference ≈ 0.03).
3. **Auto-exposure**: when you come close in a light T-shirt, the webcam darkens the whole picture. The learned background turned into a patchwork, and the shadow detector fired on half the image.

The frozen photo removes all three problems at once: no live mask, no shadows, no exposure drift. As a bonus, the empty chair makes the vanishing more convincing. The cost is that a large change in lighting while you are invisible shows when you come back; `B` retakes the photo.

### 4.2 Taking the photo

`GokuCam.capture_plate`: a 5-second countdown, then the **median of the last 15 frames**. The median removes sensor noise and someone crossing the frame briefly. The photo is then shown as a thumbnail for 4 seconds so you can check it.

An earlier version waited until the detectors saw an empty room. In practice a painting of geese and a gaming chair were detected as people, so the photo was never taken. A countdown plus a visual check is predictable; the detectors are only trusted for what they are good at.

## 5. The state machine

```
plate ──(countdown, room photo)──> visible
visible ──sign──> vanishing ──(effect done)──> hidden
hidden  ──sign──> appearing ──(effect done)──> visible
```

- `trigger(name)` picks the jutsu, draws a **seed** (random layouts such as the energy lines stay identical for the whole effect, while the per-frame `rng` makes things flicker) and calls `jutsu.start(scene, arriving)`.
- During an effect, `jutsu.render(scene, t, arriving)` is called with `t = elapsed / duration`.
- At the end of a departure, `jutsu.hidden_image(scene)` becomes the frozen image, and `jutsu.leftover_box()` records what it left behind (the log). The next arrival receives it in `scene.leftover`, so Substitution's smoke can also cover a log left by an earlier departure.
- You can leave with one jutsu and come back with another: each arrival starts from `scene.hidden`, whatever produced it.
- **Overlays** have their own small life cycle, independent of the one above: an overlay starts when its sign has been held for 0.35 s, stays active while the sign is seen (plus `linger_seconds`, 1.5 s for the mug, so hiding it for an instant does not stop the fire), and its `level` fades from 0 to 1 and back. It is drawn on top of whatever the call gets, after the jutsu. Its key switches it on regardless of the sign.

A `Scene` carries everything an effect needs: the live frame, the room photo, the current hidden image, the full-resolution person mask (and its bounding box), the seed, the per-frame random generator and the leftover box.

## 6. The effects

All effects are pure functions of the inputs and `t`, written with OpenCV and NumPy. Each one is tested on its first and last frames (the departure must start on the live frame and end **exactly** on the hidden image, so there is no jump) and on its key phases.

### 6.1 Instant Transmission (`jutsus/instant_transmission.py`)

The timing comes from frame-by-frame analysis of reference clips:

| t (departure) | Phase |
|---|---|
| 0 → 0.45 | **Energy lines**: 16 vertical lines around the person, white core and electric-blue glow, flickering, then breaking into dashes and dots |
| 0.3 → 0.55 | **Partial tearing**: 30 % of the vertical strips of the body are stretched and shifted; some become holes showing the room through you |
| 0.55 → 1 | **Full tearing**: every strip is torn, the body **gathers onto its vertical axis** into a thin column and vanishes |

Techniques:
- **Strands** (`strand_field`): the image is cut into strips 4 to 12 px wide. Each torn strip is stretched around a random anchor (`y' = anchor + (y − anchor) × stretch`) and shifted, through a `cv2.remap` y-map. Stretching rather than blurring turns dark details (hair, beard) into sharp streaks, as in the anime. The map is purely vertical, so each column stays in place.
- **Only the person tears**: the live frame **and the person mask** go through the same remap, and the result is composited over the room. The room itself is never distorted.
- **Gathering**: an x-map `x' = cx + (x − cx) / squeeze` squeezes the body onto its axis.
- **Energy lines**: adding light saturates to white on a bright wall, so the lines **tint** towards blue instead (`out × (1 − glow) + blue × glow`), then a white core is drawn on top. They stay blue on any background.
- **Arrival is not a rewind**: it plays in chronological order (lines first, then the body unfolds from a column, then a few strips settle). A rewind would put the lines at the end.

### 6.2 Substitution (`jutsus/substitution.py`)

| t (departure, 1.8 s) | Phase |
|---|---|
| 0 → 0.12 | Smoke bursts over and around the person |
| 0.08 → 0.2 | Under the smoke, the live frame cross-fades to the room photo |
| 0.15 → 0.45 | The log appears under the smoke, slightly higher and tilted, and settles |
| 0.3 → 0.9 | The smoke rises, frays into wisps and clears, revealing the log |

**Volumetric smoke** (`render_smoke`), computed on a grid at 1/4 resolution:
- 44 Gaussian puffs spread over the person, denser in the centre. They grow, rise and drift outwards, and each one fades at its own moment.
- The density is modulated by **"cauliflower" noise** (`1 − |2 × fractal noise − 1|`), which gives rounded billows instead of a uniform fog. The noise scrolls upwards over time.
- **Erosion** removes thin areas first, so the smoke frays into wisps rather than fading uniformly.
- **Lighting**: a point is darker the more smoke there is just above it (blurred density shifted down), and turbulence bumps catch the light. The colour goes from grey-blue hollows (168, 171, 177) to white tops (246, 246, 245).
- Opacity: `alpha = 1 − exp(−3 × density)`.

**Procedural log** (`make_log_sprite`), generated once per departure:
- Upright cylinder, about half the shoulder width, standing bottom-centre where the person was, top at chest height, bottom cut by the frame.
- **Bark**: fractal noise stretched 10× vertically forms vertical plates; soft dark crevices sit between them; fine grain; cylinder shading with light from the left (Lambert on the cylinder normal).
- **Cut** (seen slightly from above): irregular growth rings (rings deformed by noise and a 3-lobe wobble), a darker heart, thin radial cracks, a bark rim.
- **Integration**: the log is tinted to the room's brightness and colour cast, measured on the photo behind it, and a soft shadow is drawn behind it.

### 6.3 Peace out (`jutsus/peace_out.py`)

Imitates a heavily compressed video, as in the "bye" meme:
- The person is frozen at the moment of the sign (`Ghost`) and redrawn as **20 px blocks** (downscale + nearest-neighbour upscale), translucent: the room shows through.
- The whole picture takes a hit of pixelation while the effect plays.
- Each block dies at its own moment; blocks near the top (the head) survive longer. The survivors smear sideways row by row (`np.roll` per block row), and flicker.
- For this meme, playing the departure backwards is what looks right, so the arrival is a rewind with the live person.

### 6.4 Into the bush (`jutsus/into_the_bush.py`)

Homer Simpson backing into the hedge. The sign is one finger raised **above the forehead**: a finger at the mouth would be a "shush", and video tracking sees the Substitution's clasped hands as a single raised index, but in front of the chin.

- **Hedge**: the flat cartoon green measured on the reference clip (BGR 32, 139, 58), a few slightly darker patches and small dark leaf strokes, generated once per departure. It covers the whole width, from above the person's head down to the bottom, with a top edge made of round bumps and a jagged leafy outline (high-frequency noise added to every edge), drawn in dark green.
- **Growth** (t 0 → 0.35): the hedge rises from the bottom of the frame, behind the person.
- **Closing in** (t 0.35 → 1): each pixel of the body gets a score: its distance to the outline of the body (distance transform), plus a bonus around where the eyes usually are, plus leafy noise. The hedge covers the pixels **by rank** of that score, so the visible area shrinks steadily from the outline inwards. The thickest parts (chest, then face) remain as islands, and the face peeks out last, as in the cartoon.
- The app's person mask is padded for the other effects; here it is tightened back to the real outline, so that the hedge hugs the body without a halo.
- **Arrival**: the departure played backwards, from the hedge the call is seeing. After another jutsu's departure (for example a log), the hedge first fades in over that image.

### 6.5 This is fine (`overlays/this_is_fine.py`)

An overlay rather than a jutsu: you do not vanish, the room catches fire around you while you hold your mug.

- **Sign** (`mug_checks`): a "cup" box from the object detector, with at least 3 hand landmarks on it or within 35 % of its size around it. A mug left on the desk is not enough.
- **Flames**: 9 spots spread across the width, appearing one after the other in random order over 2.4 s, each growing in 0.5 s. A spot is a cluster of 3 to 5 **tongues**: polygons wide at the base and pointed at the tip, whose width, sway and height wobble with sines, more and more towards the tip. Every tongue is drawn three times, as in the cartoon: an orange rim, a yellow body and a pale core. Spots behind the person are taller, so they rise above the shoulders.
- **Always behind you**: the flame layer is multiplied by `1 − person mask` before compositing, so you sit in front of the fire like the dog in the meme.
- **Warm light**: the whole image, you included, is tinted towards orange with a slight flicker.
- **Smoke ceiling**: a dark band at the top with three rows of irregular puffs along its lower edge (lighter and outlined in front, as in the cartoon) that drift slowly in alternating directions and bob up and down. Only the top 20 % of the frame is processed.
- **Caption**: "This is fine." is typed at 0.08 s per letter after 1.4 s, white with a black outline, rendered with Pillow and a bold system font (OpenCV's own font is the fallback). Each prefix is laid out as the full caption would be, so the text does not shift while it is being typed, and rendered sprites are cached.
- **Cost**: about 7 ms per frame, thanks to integer blending (`cv2.blendLinear`, `cv2.transform`) instead of float arithmetic on the whole frame.

### 6.6 Absolute cinema (`overlays/absolute_cinema.py`)

The "Absolute cinema" meme: a black and white still framed by red stage curtains. The sign is two open palms (all four fingers extended) raised above the forehead, one on each side.

- **Played by its level only**: every step is a function of the overlay's fade level, never of time. While the sign is held the level rises over 1.6 s; when the hands come down (0.3 s of grace) it falls back over 1.2 s, which replays the animation backwards down to the live video.
- **Black and white** (level 0 → 0.4): the whole picture is converted to slightly contrasted grey, with a vignette.
- **Curtains** (0.15 → 0.85): two red velvet panels slide in from the sides and meet behind the person. Each panel is generated once: folds as a sum of cosines across the width, an exponent on the sheen for deep shadows, darker top and bottom, fine vertical streaks of velvet, and a dark hem on the moving edge, which is slightly wavy. The panels carry their folds as they slide.
- **Valance** (0.05 → 0.6): a velvet band with a row of swags and a gold fringe drops in from the top of the frame.
- **Title** (0.8 → 1): "ABSOLUTE CINEMA" in bold serif capitals, letter-spaced, once the curtains are closed.
- **Right behind the person**: curtains and valance are multiplied by `1 − tight mask`, the person mask brought back to the real outline (`fx.tight_mask`), so that no strip of the room shows around you.

## 7. Webcam and virtual camera

**Picking the real webcam** (`camera.py`): once OBS is installed, macOS has two cameras, and OpenCV gave index 0 to *OBS Virtual Camera*. GokuCam was reading its own output. Two safeguards:
1. `real_camera_order()` lists cameras with `system_profiler`, drops virtual ones, and sorts the rest by **unique ID**, which is the order OpenCV uses. The JSON order is different.
2. `shows_own_output()` sends a **magenta test card** to the virtual camera and checks that the opened camera does not return it. If it does, the next candidate is tried. The call may show a brief magenta flash at startup.

**Virtual camera**: `pyvirtualcam` with the OBS backend on macOS. OBS only needs to be installed and its camera extension enabled once; it does not need to run.

## 8. Performance

Measured on an Apple M1 at 1280×720:

| Step | Time per frame |
|---|---|
| Detection + state machine (hand tracking costs about as much as searching for hands) | ~30 ms |
| Still-image hand detection (only while a raised index is seen, every third frame) | ~35 ms, so ~12 ms per frame |
| Object detection (background thread, only while a hand is in view) | ~18 ms, off the main loop |
| Instant Transmission effect | ~20 ms |
| Substitution effect | ~28 ms |
| Peace out effect | ~13 ms |
| Into the bush effect | ~7 ms |
| This is fine overlay | ~7 ms |
| Absolute cinema overlay | ~14 ms |
| Log generation (once per Substitution) | ~90 ms |

Most of the time goes into MediaPipe itself; in particular, the hand landmarker looks for palms on every frame while no hand is tracked. The tricks on GokuCam's side: detection at 640×360, smoke at 1/4 resolution, blurs at reduced resolution, integer blending, slow detectors throttled or moved to a background thread, and per-effect work (log, seed, noise) done once in `start()` rather than every frame.

## 9. Privacy

- Frames are processed in memory and sent only to the virtual camera.
- Nothing is written to disk, except when you press `S`: that saves a diagnostic snapshot into `captures/`, which `.gitignore` excludes. Delete it whenever you like.
- The only network access is the download of the four MediaPipe models from `storage.googleapis.com` on first run.
