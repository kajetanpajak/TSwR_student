import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel
from scipy.integrate import solve_ivp


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.Kp = 100.0
        self.Kd = 20.0

        model_1 = ManiuplatorModel(Tp, 0.1, 0.05)
        model_2 = ManiuplatorModel(Tp, 0.01, 0.01)
        model_3 = ManiuplatorModel(Tp, 1.0, 0.3)

        self.Tp = Tp

        self.models = [model_1, model_2, model_3]
        self.i = 0

        self.prev_x = None
        self.prev_u = None

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        errors = []
        for model in self.models:
            M_inv = np.linalg.inv(model.M(self.prev_x))
            C = model.C(self.prev_x)
            
            q = self.prev_x[:2]
            q_dot = self.prev_x[2:]

            # integrate only once, prev state is the initial condition
            q_1 = q + q_dot * self.Tp
            q_dot_1 = q_dot + M_inv @ (self.prev_u - C @ q_dot) * self.Tp

            x_1 = np.concatenate([q_1, q_dot_1])

            error = np.linalg.norm(x - x_1)
            errors.append(error)
        
        self.i = np.argmin(errors)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        if self.prev_x is not None:
            self.choose_model(x)

        q = x[:2]
        q_dot = x[2:]

        q1, q2, q1_dot, q2_dot = x

        dq = (np.array([q_r[0], q_r[1]])
             - np.array([q1, q2]))
        
        dq_dot = (np.array([q_r_dot[0],q_r_dot[1]]) 
                  - np.array([q1_dot, q2_dot]))
        
        v = q_r_ddot + self.Kd * dq_dot + self.Kp * dq
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v + C @ q_dot

        # save for next model choice
        self.prev_x = x
        self.prev_u = u

        return u
