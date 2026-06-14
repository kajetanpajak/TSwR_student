import numpy as np
from .adrc_joint_controller import ADRCJointController
from .controller import Controller

from models.manipulator_model import ManiuplatorModel


class ADRController(Controller):
    def __init__(self, Tp, params):
        self.model = ManiuplatorModel(Tp, m3=0.1, r3=0.05)
        self.joint_controllers = []
        for param in params:
            self.joint_controllers.append(ADRCJointController(*param, Tp))

    def estimate_b(self, x):
        M_inv = np.linalg.inv(self.model.M(x))
        return np.diag(M_inv)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        b_hat = self.estimate_b(x)
        
        u = []
        for i, controller in enumerate(self.joint_controllers):
            u.append(controller.calculate_control([x[i], x[i+2]], q_d[i], q_d_dot[i], q_d_ddot[i], b_hat[i]))
        u = np.array(u)[:, np.newaxis]
        u = np.squeeze(u) # convert to 0-dimensional array!!
        return u
