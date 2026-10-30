import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from gokucam.hands import HOLD_SECONDS, Face, GestureTracker
from gokucam.jutsus import InstantTransmission, PeaceOut, SignInput, Substitution
from gokucam.jutsus.substitution import single_index_up

from .helpers import FACE_ABOVE, FACE_ON_HAND, joined_hands, make_hand, signs


class InstantTransmissionSignTest(unittest.TestCase):
    jutsu = InstantTransmission()

    def test_two_fingers_together_on_the_forehead(self):
        self.assertTrue(self.jutsu.detected(signs([make_hand()], FACE_ON_HAND)))

    def test_open_palm_is_rejected(self):
        self.assertFalse(self.jutsu.detected(signs([make_hand(open_palm=True)], FACE_ON_HAND)))

    def test_spread_fingers_are_rejected(self):
        self.assertFalse(self.jutsu.detected(signs([make_hand(fingers_apart=True)], FACE_ON_HAND)))

    def test_away_from_the_forehead_is_rejected(self):
        self.assertFalse(self.jutsu.detected(signs([make_hand(offset=(200, 150))], FACE_ON_HAND)))

    def test_needs_a_face(self):
        self.assertEqual(self.jutsu.checks(signs([make_hand()], None)), {"face in view": False})


class SubstitutionSignTest(unittest.TestCase):
    jutsu = Substitution()

    def test_clasped_hands_with_both_index_fingers_up(self):
        self.assertTrue(self.jutsu.detected(signs(joined_hands(), FACE_ABOVE)))
        self.assertTrue(self.jutsu.detected(signs(joined_hands(), None)))  # face hidden by the hands

    def test_real_photo_of_the_sign(self):
        """MediaPipe landmarks from a real photo: video tracking sees a single hand (index
        up), the still-image detection finds both, and the sign is recognised."""
        data = json.loads((Path(__file__).parent / "data" / "joined_hands.json").read_text())
        video = [np.float32(h) for h in data["video_hands"]]
        still = [np.float32(h) for h in data["still_hands"]]
        face = Face(np.float32(data["face"]["forehead"]), data["face"]["width"], 0.0)
        self.assertTrue(any(single_index_up(hand) for hand in video))
        self.assertFalse(self.jutsu.detected(signs(video, face)))
        self.assertTrue(self.jutsu.detected(signs(video, face, still=still)))

    def test_one_hand_is_not_enough(self):  # a "shush" gesture
        self.assertFalse(self.jutsu.detected(signs([make_hand(index_only=True)], FACE_ABOVE)))

    def test_index_fingers_apart_are_rejected(self):
        hands = [make_hand(index_only=True), make_hand(index_only=True, offset=(60, 0))]
        self.assertFalse(self.jutsu.detected(signs(hands, FACE_ABOVE)))

    def test_praying_hands_are_rejected(self):
        self.assertFalse(self.jutsu.detected(signs(joined_hands(open_palm=True), FACE_ABOVE)))

    def test_hands_on_the_forehead_are_not_the_sign(self):
        self.assertFalse(self.jutsu.detected(signs(joined_hands(), FACE_ON_HAND)))

    def test_still_image_is_only_checked_when_one_hand_raises_its_index(self):
        detect_still = mock.Mock(return_value=joined_hands())
        self.assertTrue(self.jutsu.detected(SignInput([make_hand(index_only=True)], FACE_ABOVE, detect_still)))
        detect_still.reset_mock()
        self.jutsu.detected(SignInput([make_hand(open_palm=True)], FACE_ABOVE, detect_still))
        detect_still.assert_not_called()


class PeaceOutSignTest(unittest.TestCase):
    jutsu = PeaceOut()

    def test_v_sign_away_from_the_face(self):
        self.assertTrue(self.jutsu.detected(signs([make_hand(fingers_apart=True)], FACE_ABOVE)))
        self.assertTrue(self.jutsu.detected(signs([make_hand(fingers_apart=True)], None)))

    def test_fingers_together_or_open_palm_are_not_peace(self):
        self.assertFalse(self.jutsu.detected(signs([make_hand()], FACE_ABOVE)))
        self.assertFalse(self.jutsu.detected(signs([make_hand(open_palm=True)], FACE_ABOVE)))

    def test_v_on_the_forehead_is_not_peace(self):
        self.assertFalse(self.jutsu.detected(signs([make_hand(fingers_apart=True)], FACE_ON_HAND)))

    def test_no_hand(self):
        self.assertFalse(self.jutsu.detected(signs([], FACE_ABOVE)))


def feed(tracker, detected, start, end, step=1 / 30):
    """Simulates the 30 fps video loop; returns the times at which the sign fired."""
    fired, t = [], start
    while t < end:
        if tracker.update(detected, t):
            fired.append(round(t, 2))
        t += step
    return fired


class GestureTrackerTest(unittest.TestCase):
    def test_fires_once_while_held_then_rearms_after_release(self):
        tracker = GestureTracker()
        self.assertEqual(len(feed(tracker, True, 0.0, 1.5)), 1)
        feed(tracker, False, 1.5, 2.2)
        self.assertEqual(len(feed(tracker, True, 2.2, 2.8)), 1)

    def test_hold_duration_is_respected(self):
        self.assertGreaterEqual(feed(GestureTracker(), True, 0.0, 1.0)[0], HOLD_SECONDS - 0.01)

    def test_short_dropouts_do_not_reset_the_hold(self):
        tracker = GestureTracker()
        feed(tracker, True, 0.0, 0.2)
        feed(tracker, False, 0.2, 0.3)
        self.assertEqual(feed(tracker, True, 0.3, 0.4), [0.37])


if __name__ == "__main__":
    unittest.main()
