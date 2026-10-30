import unittest

import numpy as np

from gokucam import app as appmod
from gokucam.app import GokuCam
from gokucam.jutsus import Jutsu, Scene, SignInput
from gokucam.models import DETECT_SIZE, DetectedObject, Detections
from gokucam.overlays import Overlay


class FakeDetectors:
    """No MediaPipe: an empty person mask, no face, and the hands / objects the test asks for."""

    def __init__(self):
        self.hands = []
        self.objects = []

    def process(self, small):
        return Detections(np.zeros(DETECT_SIZE[::-1], np.float32), self.hands, None, self.objects)

    def hands_on_still_image(self):
        return []


class RaisedHand(Jutsu):
    """Minimal jutsu: the sign is "a hand is visible"; the effect fades to a red room."""
    name, title, short, key, hint = "raised_hand", "Raised hand", "Hand", "h", "show a hand"
    vanish_seconds = arrival_seconds = 0.5

    def __init__(self):
        self.started = []

    def checks(self, signs: SignInput) -> dict[str, bool]:
        return {"a hand in view": bool(signs.hands)}

    def start(self, scene: Scene, arriving: bool) -> None:
        self.started.append(arriving)

    def render(self, scene: Scene, t: float, arriving: bool) -> np.ndarray:
        before, after = (scene.hidden, scene.frame) if arriving else (scene.frame, self.hidden_image(scene))
        return (before * (1 - t) + after * t).astype(np.uint8)

    def hidden_image(self, scene: Scene) -> np.ndarray:
        red = scene.plate.copy()
        red[..., 2] = 255
        return red

    def leftover_box(self):
        return (0.0, 0.0, 10.0, 10.0)


class StateMachineTest(unittest.TestCase):
    def setUp(self):
        self.detectors = FakeDetectors()
        self.jutsu = RaisedHand()
        self.cam = GokuCam(self.detectors, (320, 180), jutsus=[self.jutsu], overlays=[])
        self.frame = np.full((180, 320, 3), 120, np.uint8)
        self.now = 0.0

    def run_for(self, seconds, hand=False):
        self.detectors.hands = [np.zeros((21, 2), np.float32)] if hand else []
        out = None
        end = self.now + seconds
        while self.now < end:
            out = self.cam.step(self.frame, self.now)
            self.now += 1 / 30
        return out

    def test_full_cycle(self):
        self.run_for(appmod.PLATE_COUNTDOWN + 0.1)       # room photo
        self.assertEqual(self.cam.state, "visible")
        self.run_for(0.5, hand=True)                      # hold the sign
        self.assertEqual(self.cam.state, "vanishing")
        out = self.run_for(1.0)                           # effect plays, then hidden
        self.assertEqual(self.cam.state, "hidden")
        self.assertGreater(out[..., 2].mean(), 250)       # the call sees the jutsu's hidden image
        self.assertEqual(self.cam.leftover, (0.0, 0.0, 10.0, 10.0))
        self.run_for(0.5, hand=True)                      # sign again: arrival
        self.run_for(1.0)
        self.assertEqual(self.cam.state, "visible")
        self.assertIsNone(self.cam.leftover)
        self.assertEqual(self.jutsu.started, [False, True])

    def test_holding_the_sign_does_not_retrigger(self):
        self.run_for(appmod.PLATE_COUNTDOWN + 0.1)
        self.run_for(3.0, hand=True)                      # held through the whole effect
        self.assertEqual(self.cam.state, "hidden")        # vanished once, did not come back

    def test_debug_lines_show_missing_criteria(self):
        self.run_for(appmod.PLATE_COUNTDOWN + 0.1)
        self.assertEqual(self.cam.debug_lines(), ["Hand: missing a hand in view"])


class BlueTint(Overlay):
    """Minimal overlay: the sign is "a cup is visible"; it tints the image blue."""
    name, title, short, key, hint = "blue_tint", "Blue tint", "Blue", "u", "show a cup"
    fade_in_seconds = fade_out_seconds = 0.5
    linger_seconds = 1.0

    def __init__(self):
        self.starts = 0

    def checks(self, signs: SignInput) -> dict[str, bool]:
        return {"cup in view": any(obj.label == "cup" for obj in signs.objects)}

    def start(self, scene: Scene) -> None:
        self.starts += 1

    def render(self, image, scene, elapsed, level):
        out = image.copy()
        out[..., 0] = np.clip(image[..., 0] + 100 * level, 0, 255).astype(np.uint8)
        return out


class OverlayLifecycleTest(unittest.TestCase):
    CUP = DetectedObject("cup", 0.9, (100.0, 100.0, 140.0, 150.0))

    def setUp(self):
        self.detectors = FakeDetectors()
        self.overlay = BlueTint()
        self.cam = GokuCam(self.detectors, (320, 180), jutsus=[], overlays=[self.overlay])
        self.frame = np.full((180, 320, 3), 100, np.uint8)
        self.now = 0.0

    def run_for(self, seconds, cup=False):
        self.detectors.objects = [self.CUP] if cup else []
        out = None
        end = self.now + seconds
        while self.now < end:
            out = self.cam.step(self.frame, self.now)
            self.now += 1 / 30
        return out

    def state(self):
        return self.cam.overlay_states["blue_tint"]

    def test_fades_in_while_the_sign_lasts_and_out_after_it(self):
        self.run_for(appmod.PLATE_COUNTDOWN + 0.1)
        out = self.run_for(1.5, cup=True)
        self.assertEqual(self.state().level, 1.0)
        self.assertEqual(int(out[0, 0, 0]), 200)                      # fully drawn
        self.run_for(0.5)                                             # cup hidden for a moment
        self.assertTrue(self.state().active)                          # lingers
        out = self.run_for(2.0)
        self.assertEqual(self.state().level, 0.0)
        self.assertTrue(np.array_equal(out, self.frame))              # gone without a trace
        self.assertEqual(self.overlay.starts, 1)

    def test_key_toggles_it_without_the_sign(self):
        self.run_for(appmod.PLATE_COUNTDOWN + 0.1)
        self.cam.toggle_overlay("blue_tint", self.now)
        self.run_for(2.0)                                             # no cup, still on
        self.assertEqual(self.state().level, 1.0)
        self.cam.toggle_overlay("blue_tint", self.now)
        self.run_for(1.0)
        self.assertEqual(self.state().level, 0.0)

    def test_not_during_the_room_photo(self):
        out = self.run_for(2.0, cup=True)
        self.assertEqual(self.cam.state, "plate")
        self.assertTrue(np.array_equal(out, self.frame))


class RoomPhotoTest(unittest.TestCase):
    def setUp(self):
        self.cam = GokuCam(None, (320, 180), jutsus=[], overlays=[])
        self.room = np.full((180, 320, 3), 120, np.uint8)

    def feed(self, start, end, step=1 / 30):
        t = start
        while t < end:
            self.cam.capture_plate(self.room, t)
            t += step

    def test_photo_taken_at_the_end_of_the_countdown_whatever_the_room(self):
        self.feed(0.0, appmod.PLATE_COUNTDOWN - 0.2)
        self.assertEqual(self.cam.state, "plate")
        self.feed(appmod.PLATE_COUNTDOWN - 0.2, appmod.PLATE_COUNTDOWN + 0.1)
        self.assertEqual(self.cam.state, "visible")
        self.assertTrue(np.array_equal(self.cam.plate, self.room))
        self.assertGreater(self.cam.review_until, appmod.PLATE_COUNTDOWN)  # thumbnail shown

    def test_brief_passage_during_the_photo_is_erased(self):
        self.feed(0.0, appmod.PLATE_COUNTDOWN - 0.2)
        passing = self.room.copy()
        passing[:, 100:200] = 250  # someone crosses the frame for 3 frames
        for i in range(3):
            self.cam.capture_plate(passing, appmod.PLATE_COUNTDOWN - 0.15 + i * 0.03)
        self.feed(appmod.PLATE_COUNTDOWN - 0.05, appmod.PLATE_COUNTDOWN + 0.1)
        self.assertTrue(np.array_equal(self.cam.plate, self.room))

    def test_hidden_output_is_the_frozen_photo_with_grain(self):
        self.cam.set_plate(self.room)
        self.assertEqual(len({g.tobytes() for g in self.cam.grain}), appmod.GRAIN_FRAMES)
        for grain in self.cam.grain:
            self.assertLess(np.abs(grain.astype(np.int16) - self.room).mean(), 3)


if __name__ == "__main__":
    unittest.main()
