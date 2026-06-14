import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = np.asarray(kp)
        self.kd = np.asarray(kd)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q = np.asarray(x[:2])
        q_dot = np.asarray(x[2:])

        position_error = np.asarray(q_d) - q
        velocity_error = np.asarray(q_d_dot) - q_dot

        return self.kp * position_error + self.kd * velocity_error
