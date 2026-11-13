import unittest
from unittest.mock import patch

import numpy as np

import td_ff_live as tracker
from pose_ekf import PoseEKF


class TrackingDropoutTests(unittest.TestCase):
    def setUp(self):
        tracker.last_good_time = None
        tracker.last_good_state = {}
        tracker.pose_filter = PoseEKF(acceleration_noise=600,angular_acceleration_noise=6,
                                      gate=60,reacquire_after=0.2)
        tracker.output.fill(0)
        tracker.last_good_output.fill(0)
        self.frame = np.zeros((480,640),np.uint8)
        self.found = {10:np.zeros((4,2),np.float32),11:np.zeros((4,2),np.float32)}
        self.rv = np.float64(tracker.calibration['webcam']['rvec']).reshape(3,1)
        self.tv = np.float64(tracker.calibration['webcam']['tvec']).reshape(3,1)

    def valid_frame(self,timestamp):
        with patch.object(tracker,'detect',return_value=self.found), patch.object(
                tracker,'solve_fulcrum',return_value=((self.rv,self.tv,1.,2.),self.found)):
            return tracker.process_frame(self.frame,timestamp)

    def test_hold_fade_and_reacquire(self):
        image,state = self.valid_frame(10.)
        original = image.copy()
        self.assertTrue(original[:,:,:3].any())
        image,state = tracker.process_frame(None,10.1)
        self.assertEqual(state['tracking_status'],'holding')
        np.testing.assert_array_equal(image,original)
        image,state = tracker.process_frame(None,11.)
        self.assertEqual(state['tracking_status'],'fading')
        self.assertGreater(image[:,:,:3].sum(),0)
        self.assertLess(image[:,:,:3].sum(),original[:,:,:3].sum())
        image,state = tracker.process_frame(None,11.6)
        self.assertFalse(image[:,:,:3].any())
        image,state = self.valid_frame(12.)
        self.assertEqual(state['proj_cmd']['mode'],'colors')
        self.assertTrue(image[:,:,:3].any())

    def test_no_projection_before_first_valid_pose(self):
        image,state = tracker.process_frame(None,10.)
        self.assertFalse(image[:,:,:3].any())
        self.assertEqual(state['proj_cmd']['mode'],'lost')


if __name__=='__main__':
    unittest.main()
