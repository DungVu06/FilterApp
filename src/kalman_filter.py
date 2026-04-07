import numpy as np

class KalmanFilter:
    def __init__(self, q=1e-4, r=1e-2):
        """
        Q: Process noise - Càng nhỏ thì đường đi càng mượt (nhưng phản ứng chậm)
        R: Measurement noise - Càng lớn thì càng tin vào dự đoán, ít tin vào camera (giảm rung)
        """
        self.q = q 
        self.r = r
        self.p = 1.0
        self.x = None

    def smooth(self, prediction, measurement):
        if self.x is None:
            self.x = measurement
            return self.x
        self.x = prediction
        self.p = self.p + self.q

        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p

        return self.x