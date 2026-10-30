import json
import unittest
from unittest import mock

import numpy as np

from gokucam import camera


class FakeVirtualCamera:
    width, height = 64, 36

    def __init__(self):
        self.last = None

    def send(self, frame):
        self.last = frame


class FakeCapture:
    """Returns either a real-looking frame or what the virtual camera receives (loopback)."""

    def __init__(self, virtual=None):
        self.virtual = virtual

    def read(self):
        if self.virtual is not None and self.virtual.last is not None:
            return True, self.virtual.last.copy()
        return True, np.full((36, 64, 3), 120, np.uint8)


class CameraChoiceTest(unittest.TestCase):
    # system_profiler on a Mac with OBS: the JSON order is NOT OpenCV's order
    PROFILER = {"SPCameraDataType": [
        {"_name": "Caméra FaceTime HD", "spcamera_unique-id": "EAB7A68F-EC2B-4487-AADF-D8A91C1CB782"},
        {"_name": "OBS Virtual Camera", "spcamera_unique-id": "7626645E-4425-469E-9D8B-97E0FA59AC75"},
    ]}

    def test_real_camera_gets_its_opencv_index_and_the_virtual_one_is_skipped(self):
        result = mock.Mock(stdout=json.dumps(self.PROFILER))
        with mock.patch.object(camera.platform, "system", return_value="Darwin"), \
                mock.patch.object(camera.subprocess, "run", return_value=result):
            self.assertEqual(camera.real_camera_order(), [(1, "Caméra FaceTime HD")])

    def test_virtual_camera_names(self):
        self.assertTrue(camera.is_virtual_camera("OBS Virtual Camera"))
        self.assertTrue(camera.is_virtual_camera("Snap Camera Virtual"))
        self.assertFalse(camera.is_virtual_camera("FaceTime HD Camera"))

    def test_own_output_is_recognised(self):
        virtual = FakeVirtualCamera()
        self.assertTrue(camera.shows_own_output(FakeCapture(virtual), virtual))
        self.assertFalse(camera.shows_own_output(FakeCapture(), FakeVirtualCamera()))


if __name__ == "__main__":
    unittest.main()
