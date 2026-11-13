import unittest

import cv2
import numpy as np

from pose_ekf import PoseEKF


class PoseEKFTests(unittest.TestCase):
    def test_stationary_noise_is_reduced(self):
        rng = np.random.default_rng(12)
        ekf = PoseEKF()
        raw,filtered = [],[]
        for i in range(300):
            tv = np.array([10.,20.,600.])+rng.normal(0,2,3)
            rv = rng.normal(0,0.015,3)
            (r,t),accepted = ekf.update(rv,tv,i/12)
            self.assertTrue(accepted)
            if i>30:
                raw.append(tv)
                filtered.append(t.ravel())
            self.assertGreaterEqual(np.linalg.eigvalsh(ekf.covariance).min(),-1e-9)
        self.assertLess(np.std(filtered,axis=0).mean(),np.std(raw,axis=0).mean()*0.85)

    def test_rotation_wrap_is_not_a_jump(self):
        ekf = PoseEKF()
        ekf.update([0,0,np.pi-0.01],[0,0,600],0)
        (rv,_),accepted = ekf.update([0,0,-np.pi+0.01],[0,0,600],0.08)
        self.assertTrue(accepted)
        R = cv2.Rodrigues(rv)[0]
        self.assertAlmostEqual(np.linalg.det(R),1.0)
        self.assertLess(np.linalg.norm(cv2.Rodrigues(R.T @ cv2.Rodrigues(np.array([0.,0.,np.pi]))[0])[0]),0.02)

    def test_constant_velocity(self):
        ekf = PoseEKF()
        for i in range(100):
            (_,tv),accepted = ekf.update([0,0,i*0.01],[i*2.,0,600],i/12)
            self.assertTrue(accepted)
        self.assertLess(abs(tv[0,0]-198),1)
        self.assertAlmostEqual(ekf.velocity[0],24,delta=1)

    def test_spike_is_rejected(self):
        ekf = PoseEKF()
        for i in range(30):
            ekf.update([0,0,0],[0,0,600],i/12)
        (_,tv),accepted = ekf.update([1,0,0],[500,0,600],30/12)
        self.assertFalse(accepted)
        self.assertLess(abs(tv[0,0]),1)
        _,accepted = ekf.update([0,0,0],[0,0,600],31/12)
        self.assertTrue(accepted)

    def test_live_filter_reacquires_quickly_after_real_move(self):
        ekf = PoseEKF(acceleration_noise=600,angular_acceleration_noise=6,gate=60,reacquire_after=0.2)
        for i in range(30):
            ekf.update([0,0,0],[0,0,600],i/30)
        accepted = False
        for i in range(30,38):
            (_,tv),accepted = ekf.update([0,0,0.5],[100,0,600],i/30)
            if accepted:
                break
        self.assertTrue(accepted)
        self.assertLessEqual(i,37)
        self.assertAlmostEqual(tv[0,0],100,delta=3)

    def test_stillness_lock_suppresses_position_and_rotation_jitter(self):
        ekf = PoseEKF(stationary_enabled=True,acceleration_noise=600,angular_acceleration_noise=6)
        rng = np.random.default_rng(27)
        positions,rotations = [],[]
        for i in range(240):
            rv = rng.normal(0,0.003,3)
            tv = np.array([0.,0.,600.])+rng.normal(0,[0.6,0.5,1.0],3)
            (r,t),accepted = ekf.update(rv,tv,i/30)
            if i>60:
                positions.append(t.ravel())
                rotations.append(r.ravel())
        self.assertTrue(ekf.stationary)
        self.assertLess(np.std(positions,axis=0).max(),0.15)
        self.assertLess(np.std(rotations,axis=0).max(),0.001)

    def test_stillness_releases_after_two_motion_frames_not_one_spike(self):
        ekf = PoseEKF(stationary_enabled=True)
        for i in range(30):
            ekf.update([0,0,0],[0,0,600],i/30)
        self.assertTrue(ekf.stationary)
        (_,tv),_ = ekf.update([0,0,0],[20,0,600],1.0)
        self.assertTrue(ekf.stationary)
        self.assertEqual(tv[0,0],0)
        ekf.update([0,0,0],[0,0,600],31/30)
        self.assertTrue(ekf.stationary)
        ekf.update([0,0,0.1],[20,0,600],32/30)
        (_,tv),accepted = ekf.update([0,0,0.1],[21,0,600],33/30)
        self.assertTrue(accepted)
        self.assertFalse(ekf.stationary)
        self.assertAlmostEqual(tv[0,0],21,delta=3)

    def test_locked_pose_follows_small_offset(self):
        ekf = PoseEKF(stationary_enabled=True)
        for i in range(30):
            ekf.update([0,0,0],[0,0,600],i/30)
        self.assertTrue(ekf.stationary)
        rng = np.random.default_rng(7)
        for i in range(30,120):
            position = np.array([10.,0.,600.])+rng.normal(0,[0.8,0.7,2.5],3)
            (rv,tv),accepted = ekf.update(rng.normal(0,0.005,3),position,i/30)
            self.assertTrue(accepted)
            self.assertTrue(ekf.stationary)
        self.assertGreater(tv[0,0],6)

    def test_stillness_remains_locked_with_depth_noise(self):
        ekf = PoseEKF(stationary_enabled=True)
        for i in range(30):
            ekf.update([0,0,0],[0,0,600],i/30)
        rng = np.random.default_rng(42)
        for i in range(30,180):
            position = np.array([0.,0.,600.])+rng.normal(0,[0.8,0.7,2.5],3)
            (rv,tv),accepted = ekf.update(rng.normal(0,0.007,3),position,i/30)
            self.assertTrue(accepted)
            self.assertTrue(ekf.stationary)
            np.testing.assert_allclose(tv.ravel(),[0,0,600],atol=1.5)

    def test_stillness_does_not_lock_continuous_motion(self):
        ekf = PoseEKF(stationary_enabled=True)
        for i in range(90):
            ekf.update([0,0,i*0.01],[i*0.8,0,600],i/30)
            self.assertFalse(ekf.stationary)

    def test_reacquisition_resets_after_gap(self):
        ekf = PoseEKF()
        ekf.update([0,0,0],[0,0,600],0)
        (_,tv),accepted = ekf.update([0,1,0],[300,0,600],1)
        self.assertTrue(accepted)
        self.assertEqual(tv[0,0],300)


if __name__=='__main__':
    unittest.main()
