from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        z = np.reshape(self.state, (len(self.state), 1))
        u = np.reshape(u, (1, 1))
        z_dot = self.A @ z + self.B @ u + self.L @ (q - self.W @ z)
        self.state = self.state + np.reshape(z_dot * self.Tp, (1, len(self.state)))
        self.state = self.state[0]

    def get_state(self):
        return self.state
    
def main():
    eso = ESO(A=np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), B=np.array([[0], [1], [0]]), W=np.array([[1, 0, 0]]), L=np.array([[10], [100], [20]]), state=[0, 0, 0], Tp=0.01)
    print(eso.state.shape)

    eso.update(1, 1)
    print(eso.state)


if __name__ == "__main__":
    main()
