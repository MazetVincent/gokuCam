import unittest

import numpy as np

from gokucam.jutsus import Scene, default_jutsus
from gokucam.models import DetectedObject
from gokucam.overlays import AbsoluteCinema, ThisIsFine
from gokucam.overlays.this_is_fine import CAPTION_CHAR, CAPTION_START, mug_checks

from .helpers import FACE_ABOVE, FACE_BELOW, make_hand, signs


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


class AbsoluteCinemaSignTest(unittest.TestCase):
    overlay = AbsoluteCinema()

    @staticmethod
    def palms(apart=150, **kwargs):
        return [make_hand(open_palm=True, **kwargs), make_hand(open_palm=True, offset=(apart, 0), **kwargs)]

    def test_both_open_palms_raised_above_the_head(self):
        self.assertTrue(all(self.overlay.checks(signs(self.palms(), FACE_BELOW)).values()))

    def test_one_hand_hands_together_low_hands_or_closed_hands_are_not_the_sign(self):
        for hands, face in (([make_hand(open_palm=True)], FACE_BELOW),     # one hand
                            (self.palms(apart=20), FACE_BELOW),           # both on the same side
                            (self.palms(), FACE_ABOVE),                   # below the head
                            ([make_hand(), make_hand(offset=(150, 0))], FACE_BELOW)):  # not open
            self.assertFalse(all(self.overlay.checks(signs(hands, face)).values()))

    def test_does_not_trigger_a_jutsu(self):
        raised = signs(self.palms(), FACE_BELOW)
        self.assertFalse(any(jutsu.detected(raised) for jutsu in default_jutsus()))


class AbsoluteCinemaRenderTest(unittest.TestCase):
    def setUp(self):
        h, w = 360, 640
        self.frame = np.zeros((h, w, 3), np.uint8)
        self.frame[:] = (40, 160, 90)                     # a green room...
        self.mask = np.zeros((h, w), np.float32)
        self.mask[100:, 250:390] = 1.0
        self.frame[100:, 250:390] = (200, 120, 60)       # ...and a blue person
        self.scene = Scene(frame=self.frame, plate=self.frame, hidden=self.frame, mask=self.mask,
                           seed=1, rng=np.random.default_rng(0))
        self.cinema = AbsoluteCinema()
        self.cinema.start(self.scene)

    def render(self, level, elapsed=0.0):
        return self.cinema.render(self.frame, self.scene, elapsed, level)

    @staticmethod
    def red(region):
        """Red-dominant pixels, including the dark red deep in the folds."""
        b, g, r = (region[..., i].astype(int) for i in range(3))
        return (r > 1.8 * g) & (r > 1.8 * b) & (r > 15)

    @staticmethod
    def saturation(region):
        return int(region.max(axis=2).astype(int).mean() - region.min(axis=2).astype(int).mean())

    def test_nothing_at_level_zero(self):
        self.assertTrue(np.array_equal(self.render(0.0), self.frame))

    def test_turns_black_and_white_first(self):
        out = self.render(0.4)
        self.assertLess(self.saturation(out[150:300, 280:360]), 5)   # the person is grey
        self.assertFalse(self.red(out[150:300, 300:340]).any())      # no curtain in the middle yet

    def test_curtains_close_behind_the_person(self):
        out = self.render(1.0)
        self.assertGreater(self.red(out[150:330, :200]).mean(), 0.95)    # room hidden by the curtains
        self.assertGreater(self.red(out[150:330, 440:]).mean(), 0.95)
        self.assertFalse(self.red(out[200:300, 280:360]).any())          # the person stays in front

    def test_title_appears_once_the_curtains_are_closed(self):
        def white(out):
            return int((out[int(0.75 * 360):].min(axis=2) > 230).sum())
        self.assertEqual(white(self.render(0.7)), 0)
        self.assertGreater(white(self.render(1.0)), 200)

    def test_depends_only_on_the_level_so_lowering_the_hands_plays_it_backwards(self):
        for level in (0.2, 0.5, 0.9):
            self.assertTrue(np.array_equal(self.render(level, elapsed=0.3), self.render(level, elapsed=5.0)))


if __name__ == "__main__":
    unittest.main()
