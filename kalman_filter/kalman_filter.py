""" with great help of https://www.cs.utexas.edu/~teammco/misc/kalman_filter/"""

import numpy as np
class KalmanFilter:

    def __init__(self, X, A, Q, Z, H, R, P, B=np.array([0]), C=np.array([0])):
        """
        Initialise the filter
        Args:
            X: State
            P: Covariance
            A: State Transition
            B: Control matrix (zeros in our case)
            C: Control vector (zeros in our case)
            Q: Action Uncertainty
            Z: Measurement
            H: Measurement matrix
            R: Sensor Noise
        """
        self.X = X
        self.P = P
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.Z = Z
        self.H = H
        self.R = R

    def predict(self):
        """
        Predict the future state of system

        x = (A * x) + (B * c)
        P = (A * P * AT) + Q       ← AT is the matrix transpose of A
        """
        self.X = np.dot(self.A, self.X) + np.dot(self.B, self.C)
        self.P = self.A.dot(self.P).dot(self.A.T) + self.Q

        return self.X

    def update(self, m):
        """
        Update the Kalman Filter from a measurement
        Correction Step:
        S = (H * P * HT) + R       ← HT is the matrix transpose of H
        K = P * HT * S-1           ← S-1 is the matrix inverse of S
        y = m - (H * x)
        x = x + (K * y)
        P = (I - (K * H)) * P      ← I is the Identity matrix
        """
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        y = m - np.dot(self.H, self.X)
        self.X = self.X + np.dot(K, y)
        self.P = self.P - K.dot(self.H).dot(self.P)

        return self.X
