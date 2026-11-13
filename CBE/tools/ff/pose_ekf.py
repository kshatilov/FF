import cv2
import numpy as np


def skew(v):
    x,y,z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],dtype=float)


class PoseEKF:
    def __init__(self, position_noise=2.0, rotation_noise=0.015,
                 acceleration_noise=150.0, angular_acceleration_noise=2.5,
                 gate=30.0, reacquire_after=0.5, stationary_enabled=False):
        self.measurement_noise = np.diag([position_noise**2]*3+[rotation_noise**2]*3)
        self.acceleration_noise = acceleration_noise
        self.angular_acceleration_noise = angular_acceleration_noise
        self.gate = gate
        self.reacquire_after = reacquire_after
        self.stationary_enabled = stationary_enabled
        self.stationary = False
        self.history = []
        self.recent_motion = []
        self.locked_window = []
        self.time = None
        self.accepted_time = None
        self.rejections = 0
        self.innovation_squared = 0.0

    def reset(self, rv, tv, timestamp):
        self.position = np.float64(tv).reshape(3).copy()
        self.rotation = cv2.Rodrigues(np.float64(rv).reshape(3))[0]
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.covariance = np.diag([4.0]*3+[10000.0]*3+[0.000225]*3+[4.0]*3)
        self.stationary = False
        self.history = []
        self.recent_motion = []
        self.locked_window = []
        self.time = self.accepted_time = timestamp
        self.rejections = 0
        self.innovation_squared = 0.0

    def stillness(self, rv, tv, timestamp):
        position = np.float64(tv).reshape(3)
        rotation = cv2.Rodrigues(np.float64(rv).reshape(3))[0]
        if self.stationary:
            distance = np.linalg.norm(position-self.position)
            angle = np.linalg.norm(cv2.Rodrigues(self.rotation.T @ rotation)[0])
            self.recent_motion.append(max(distance/18.0,angle/0.08))
            del self.recent_motion[:-3]
            if len(self.recent_motion)>=2 and np.median(self.recent_motion)>1.0:
                self.stationary = False
                self.recent_motion = []
                self.locked_window = []
                return 'released'
            self.locked_window.append((timestamp,position.copy(),rotation))
            self.locked_window = [s for s in self.locked_window if timestamp-s[0]<=0.8]
            if len(self.locked_window)>=6:
                positions = np.array([s[1] for s in self.locked_window])
                delta = np.median(positions,axis=0)-self.position
                if np.linalg.norm(delta)>1.0:
                    self.position += delta*0.35
                reference = self.locked_window[0][2]
                vectors = np.array([cv2.Rodrigues(reference.T @ s[2])[0].ravel() for s in self.locked_window])
                target = reference @ cv2.Rodrigues(np.median(vectors,axis=0))[0]
                correction = cv2.Rodrigues(self.rotation.T @ target)[0].ravel()
                if np.linalg.norm(correction)>0.003:
                    self.rotation = self.rotation @ cv2.Rodrigues(correction*0.35)[0]
            return 'locked'
        self.history.append((timestamp,position.copy(),rotation))
        self.history = [s for s in self.history if timestamp-s[0]<=0.4]
        if len(self.history)<8 or timestamp-self.history[0][0]<0.3:
            return 'moving'
        positions = np.array([s[1] for s in self.history])
        center = np.median(positions,axis=0)
        reference = self.history[0][2]
        rotations = np.array([cv2.Rodrigues(reference.T @ s[2])[0].ravel() for s in self.history])
        rotation_center = np.median(rotations,axis=0)
        split = len(self.history)//2
        drift = np.linalg.norm(np.median(positions[:split],axis=0)-np.median(positions[split:],axis=0))
        angle_drift = np.linalg.norm(np.median(rotations[:split],axis=0)-np.median(rotations[split:],axis=0))
        if (np.percentile(np.linalg.norm(positions-center,axis=1),90)<16
                and np.percentile(np.linalg.norm(rotations-rotation_center,axis=1),90)<0.05
                and drift<3 and angle_drift<0.015):
            self.position = center
            self.rotation = reference @ cv2.Rodrigues(rotation_center)[0]
            self.velocity.fill(0)
            self.angular_velocity.fill(0)
            self.stationary = True
            self.recent_motion = []
            self.locked_window = []
            return 'locked'
        return 'moving'

    def update(self, rv, tv, timestamp, noise_scale=1.0, process_scale=1.0):
        if self.time is None or timestamp-self.accepted_time > self.reacquire_after or timestamp < self.time:
            self.reset(rv,tv,timestamp)
            return self.pose(), True
        if self.stationary_enabled:
            mode = self.stillness(rv,tv,timestamp)
            if mode=='released':
                self.reset(rv,tv,timestamp)
                return self.pose(),True
            if mode=='locked':
                self.time = self.accepted_time = timestamp
                self.innovation_squared = 0.0
                self.rejections = 0
                return self.pose(),True
        dt = min(max(timestamp-self.time,0.0001),0.25)
        self.time = timestamp
        self.position += self.velocity*dt
        increment = cv2.Rodrigues(self.angular_velocity*dt)[0]
        self.rotation = self.rotation @ increment
        F = np.eye(12)
        F[0:3,3:6] = np.eye(3)*dt
        F[6:9,6:9] = increment.T
        F[6:9,9:12] = np.eye(3)*dt
        Q = np.zeros((12,12))
        process_scale = max(0.05,float(process_scale))
        for p,v,noise in ((0,3,self.acceleration_noise),(6,9,self.angular_acceleration_noise)):
            G = np.zeros((12,3))
            G[p:p+3] = np.eye(3)*dt**2/2
            G[v:v+3] = np.eye(3)*dt
            Q += G @ G.T * (noise*process_scale)**2
        self.covariance = F @ self.covariance @ F.T + Q
        measured_rotation = cv2.Rodrigues(np.float64(rv).reshape(3))[0]
        residual = np.r_[np.float64(tv).reshape(3)-self.position,
                         cv2.Rodrigues(self.rotation.T @ measured_rotation)[0].reshape(3)]
        H = np.zeros((6,12))
        H[:3,:3] = np.eye(3)
        H[3:,6:9] = np.eye(3)
        R = self.measurement_noise*max(1.0,float(noise_scale))**2
        S = H @ self.covariance @ H.T + R
        self.innovation_squared = float(residual @ np.linalg.solve(S,residual))
        if self.innovation_squared > self.gate:
            self.rejections += 1
            return self.pose(), False
        gain = np.linalg.solve(S,H @ self.covariance).T
        correction = gain @ residual
        self.position += correction[:3]
        self.velocity += correction[3:6]
        self.rotation = self.rotation @ cv2.Rodrigues(correction[6:9])[0]
        self.angular_velocity += correction[9:12]
        A = np.eye(12)-gain @ H
        self.covariance = A @ self.covariance @ A.T + gain @ R @ gain.T
        reset = np.eye(12)
        reset[6:9,6:9] -= skew(correction[6:9])/2
        self.covariance = reset @ self.covariance @ reset.T
        self.covariance = (self.covariance+self.covariance.T)/2
        self.accepted_time = timestamp
        self.rejections = 0
        return self.pose(), True

    def predict(self, timestamp):
        if self.time is None:
            return None
        dt = min(max(timestamp-self.time,0.0),0.25)
        self.time = timestamp
        if dt>0:
            self.position += self.velocity*dt
            increment = cv2.Rodrigues(self.angular_velocity*dt)[0]
            self.rotation = self.rotation @ increment
            F = np.eye(12)
            F[0:3,3:6] = np.eye(3)*dt
            F[6:9,6:9] = increment.T
            F[6:9,9:12] = np.eye(3)*dt
            Q = np.zeros((12,12))
            for p,v,noise in ((0,3,self.acceleration_noise),(6,9,self.angular_acceleration_noise)):
                G = np.zeros((12,3))
                G[p:p+3] = np.eye(3)*dt**2/2
                G[v:v+3] = np.eye(3)*dt
                Q += G @ G.T * noise**2
            self.covariance = F @ self.covariance @ F.T + Q
        return self.pose()

    def pose(self):
        return cv2.Rodrigues(self.rotation)[0],self.position.reshape(3,1).copy()
