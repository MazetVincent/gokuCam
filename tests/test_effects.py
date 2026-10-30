import unittest

import numpy as np

from gokucam.jutsus import peace_out as po
from gokucam.jutsus import substitution as sub
from gokucam.jutsus.instant_transmission import render_teleport, tear_into_strands


class InstantTransmissionEffectTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((180, 320, 3), 200, np.uint8)
        self.plate = np.full((180, 320, 3), 30, np.uint8)
        self.plate[85:95] = 90  # some horizontal detail in the room
        self.mask = np.zeros((180, 320), np.float32)
        self.mask[40:160, 120:200] = 1.0  # the person
        self.rng = np.random.default_rng(0)

    def render(self, t, arriving=False):
        return render_teleport(self.frame, self.plate, self.mask, t, arriving, 7, self.rng)

    @staticmethod
    def blue_columns(out):
        """Columns crossed by an energy line (clearly bluer than red)."""
        return np.nonzero(((out[:, :, 0].astype(int) - out[:, :, 2]) > 60).any(axis=0))[0]

    def test_departure_and_arrival_end_exactly(self):
        self.assertTrue(np.array_equal(self.render(0.0), self.frame))
        self.assertTrue(np.array_equal(self.render(1.0), self.plate))
        self.assertTrue(np.array_equal(self.render(0.0, arriving=True), self.plate))
        self.assertTrue(np.array_equal(self.render(1.0, arriving=True), self.frame))

    def test_blue_lines_open_both_departure_and_arrival_around_the_person(self):
        for arriving in (False, True):
            columns = self.blue_columns(self.render(0.15, arriving))
            self.assertGreater(columns.size, 5)
            self.assertTrue(((columns > 100) & (columns < 220)).all())
            # the arrival is not a rewind: no blue lines at the end
            self.assertEqual(self.blue_columns(self.render(0.85, arriving)).size, 0)

    def test_partial_phase_tears_some_strips_and_shows_the_room_through(self):
        body = self.render(0.47)[60:140, 120:200, 0]
        self.assertGreater(int((body == 200).all(axis=0).sum()), 10)  # intact strips
        room = (body == self.plate[60:140, 120:200, 0]).all(axis=0)
        self.assertGreater(int(room.sum()), 1)                        # holes

    def test_final_phase_gathers_the_body_on_its_axis(self):
        lit = np.nonzero((self.render(0.92)[:, :, 0] > 100).any(axis=0))[0]
        self.assertGreater(lit.size, 0)
        self.assertLess(lit.size, 20)  # 80 columns wide at the start
        self.assertTrue(((lit > 140) & (lit < 180)).all())

    def test_the_room_is_never_torn(self):
        for t in (0.5, 0.65, 0.8, 0.95):
            out = self.render(t)
            self.assertTrue(np.array_equal(out[:, :110], self.plate[:, :110]))
            self.assertTrue(np.array_equal(out[:, 210:], self.plate[:, 210:]))

    def test_tear_turns_horizontal_detail_into_vertical_strands(self):
        img = np.zeros((180, 320, 3), np.uint8)
        img[85:95] = 255
        lit_rows = (tear_into_strands(img, 1.0, self.rng)[:, :, 0] > 128).sum(axis=0)
        self.assertGreater(lit_rows.max(), 40)           # long streaks
        self.assertGreater(len(np.unique(lit_rows)), 5)  # different from strip to strip
        self.assertTrue(np.array_equal(tear_into_strands(img, 0.0, self.rng), img))


class SubstitutionEffectTest(unittest.TestCase):
    SIZE = (640, 360)

    def setUp(self):
        self.frame = np.full((360, 640, 3), 200, np.uint8)
        self.plate = np.zeros((360, 640, 3), np.uint8)  # black room: only the log and smoke show up
        self.box = (220.0, 80.0, 420.0, 360.0)          # the person
        self.layout = sub.log_layout(self.box, self.SIZE)
        self.smoke = sub.make_smoke(self.box, 3, sub.VANISH_FADE, self.SIZE)

    def render(self, t, frame=None):
        return sub.render_substitution(self.frame if frame is None else frame, self.plate, self.layout, self.smoke, t)

    def test_starts_live_and_ends_with_the_log_at_rest(self):
        self.assertTrue(np.array_equal(self.render(0.0), self.frame))
        rest = sub.log_at_rest(self.plate, self.layout)
        self.assertTrue(np.array_equal(self.render(1.0), rest))
        self.assertTrue(np.allclose(self.render(0.97), rest, atol=3))  # smoke gone, no jump

    def test_log_stands_upright_at_the_bottom_where_the_person_was(self):
        rows, cols = np.nonzero(sub.log_at_rest(self.plate, self.layout).max(axis=2) > 20)
        width, height = cols.max() - cols.min(), rows.max() - rows.min()
        self.assertGreater(height, width)                                     # upright
        self.assertEqual(rows.max(), 359)                                     # cut by the frame
        self.assertAlmostEqual((cols.min() + cols.max()) / 2, 320, delta=10)  # centred on the person
        self.assertGreater(width, 0.22 * 640)                                 # big
        self.assertGreater(rows.min(), 0.25 * 360)                            # top at chest height

    def test_smoke_hides_the_person_during_the_swap(self):
        out = self.render(0.15, frame=np.zeros_like(self.frame))
        self.assertGreater(out[150:300, 280:360].mean(), 120)  # dense, bright smoke in the centre

    def test_arrival_swaps_the_log_for_the_live_image(self):
        hidden = sub.log_at_rest(self.plate, self.layout)
        smoke = sub.make_smoke(self.box, 4, sub.ARRIVAL_FADE, self.SIZE)
        self.assertTrue(np.array_equal(sub.render_smoke_arrival(self.frame, hidden, smoke, 0.0), hidden))
        self.assertTrue(np.array_equal(sub.render_smoke_arrival(self.frame, hidden, smoke, 1.0), self.frame))


class PeaceOutEffectTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((360, 640, 3), 200, np.uint8)
        self.plate = np.full((360, 640, 3), 40, np.uint8)
        self.plate[::7] = 90
        self.mask = np.zeros((360, 640), np.float32)
        self.mask[60:360, 240:400] = 1.0
        self.ghost = po.Ghost(self.frame.copy(), self.mask, 5)
        self.rng = np.random.default_rng(0)

    def render(self, t):
        return po.render_peace_out(self.frame, self.plate, self.ghost, t, self.rng)

    def test_starts_live_and_ends_on_the_empty_room(self):
        self.assertTrue(np.array_equal(self.render(0.0), self.frame))
        self.assertTrue(np.array_equal(self.render(1.0), self.plate))

    def test_the_person_becomes_a_translucent_ghost(self):
        body = self.render(0.3)[120:300, 260:380, 0].astype(int)
        self.assertGreater(body.mean(), 60)   # still visible...
        self.assertLess(body.mean(), 190)     # ...but the room shows through

    def test_only_a_few_blocks_remain_before_vanishing(self):
        def person_pixels(t):
            return int((self.render(t)[:, 240:400, 0] > 120).sum())
        self.assertGreater(person_pixels(0.3), 2 * person_pixels(0.7))
        self.assertEqual(person_pixels(0.95), 0)

    def test_arrival_ends_on_the_live_image(self):
        self.assertTrue(np.array_equal(po.render_peace_in(self.frame, self.plate, self.mask, 0.0, 5, self.rng), self.plate))
        self.assertTrue(np.array_equal(po.render_peace_in(self.frame, self.plate, self.mask, 1.0, 5, self.rng), self.frame))


if __name__ == "__main__":
    unittest.main()
