import math
import time

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=0.5, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, t, x):
        dt = t - self.t_prev
        if dt <= 0: return self.x_prev
        dx = (x - self.x_prev) / dt
        edx = self.dx_prev + (1.0 / (1.0 + (1.0 / (2 * 3.14159 * self.d_cutoff)) / dt)) * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = 1.0 / (1.0 + (1.0 / (2 * 3.14159 * cutoff)) / dt)
        x_filtered = self.x_prev + alpha * (x - self.x_prev)
        self.t_prev, self.x_prev, self.dx_prev = t, x_filtered, edx
        return x_filtered