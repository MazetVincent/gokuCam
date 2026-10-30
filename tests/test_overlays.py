import unittest

import numpy as np

from gokucam.jutsus import Scene
from gokucam.models import DetectedObject
from gokucam.overlays import ThisIsFine
from gokucam.overlays.this_is_fine import CAPTION_CHAR, CAPTION_START, mug_checks

from .helpers import make_hand, signs


class MugSignTest(unittest.TestCase):
    MUG = DetectedObject("cup", 0.8, (280.0, 170.0, 330.0, 230.0))  # around the synthetic hand's fingers

    def checks(self, objects, hands):
        s = signs(hands)
        s.objects = objects
        return mug_checks(s)

    def test_mug_held_in_hand(self):
        self.assertTrue(all(self.checks([self.MUG], [make_hand()]).values()))

    def test_mug_on_the_desk_is_not_enough(self):
        far = DetectedObject("cup", 0.8, (20.0, 20.0, 60.0, 70.0))
        self.assertEqual(self.checks([far], [make_hand()]), {"mug in view": True, "hand on the mug": False})

    def test_no_mug(self):
        self.assertEqual(self.checks([], [make_hand()]), {"mug in view": False})


class ThisIsFineRenderTest(unittest.TestCase):
    def setUp(self):
        h, w = 360, 640
        self.frame = np.full((h, w, 3), 90, np.uint8)
        self.mask = np.zeros((h, w), np.float32)
        self.mask[80:, 240:400] = 1.0  # the person
        self.scene = Scene(frame=self.frame, plate=self.frame, hidden=self.frame, mask=self.mask,
                           seed=2, rng=np.random.default_rng(0))
        self.fine = ThisIsFine()
        self.fine.start(self.scene)

    def render(self, t, level=1.0, image=None):
        return self.fine.render(self.frame if image is None else image, self.scene, t, level)

    @staticmethod
    def fire_pixels(out):
        """Flame-coloured pixels: much more red than blue."""
        return (out[..., 2].astype(int) - out[..., 0]) > 80

    def test_nothing_at_level_zero(self):
        self.assertTrue(np.array_equal(self.render(3.0, level=0.0), self.frame))

    def test_flames_appear_one_after_the_other(self):
        counts = [int(self.fire_pixels(self.render(t)).sum()) for t in (0.3, 1.2, 3.0)]
        self.assertLess(counts[0], counts[1])
        self.assertLess(counts[1], counts[2])

    def test_flames_stay_behind_the_person(self):
        fire = self.fire_pixels(self.render(3.0))
        self.assertGreater(int(fire[:, :200].sum()), 1000)        # around the person
        self.assertEqual(int(fire[150:300, 270:370].sum()), 0)    # never on the person

    def test_smoke_ceiling_at_the_top(self):
        bright = np.full_like(self.frame, 230)
        out = self.render(3.0, image=bright)
        self.assertLess(out[:20].mean(), 120)                      # dark smoke at the top

    def test_caption_is_typed_letter_by_letter(self):
        white = lambda out: int((out[int(0.75 * 360):int(0.9 * 360)].min(axis=2) > 240).sum())  # noqa: E731
        before = white(self.render(CAPTION_START - 0.1))
        few = white(self.render(CAPTION_START + 3 * CAPTION_CHAR))
        full = white(self.render(CAPTION_START + 20 * CAPTION_CHAR))
        self.assertEqual(before, 0)
        self.assertGreater(few, 0)
        self.assertGreater(full, 2 * few)


if __name__ == "__main__":
    unittest.main()
