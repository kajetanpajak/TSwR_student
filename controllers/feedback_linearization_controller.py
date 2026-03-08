import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)
        self.Kp = 20.0
        self.Kd = 10.0

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """

        M = self.model.M(x)
        C = self.model.C(x)

        q1, q2, q1_dot, q2_dot = x
        
        dq = (np.array([q_r[0], q_r[1]])
             - np.array([q1, q2]))
        
        dq_dot = (np.array([q_r_dot[0],q_r_dot[1]]) 
                  - np.array([q1_dot, q2_dot]))
        
        v = np.array([q_r_ddot[0], # v = q_ddot + Kd(q_dotr - q_dot) + Kp(qr - q)
                      q_r_ddot[1]]) + self.Kd * dq_dot + self.Kp * dq
    
        q_dot = np.array([q1_dot, q2_dot])

        tau = M @ v + C @ q_dot

        return tau

        # return NotImplementedError()
