import numpy as np
import torch
from pytorch_dmpling.cs import CanonicalSystem
from pytorch_dmpling import utils

class DMP:
    def __init__(self, T, dt, a=150, b=25, w=None, device=torch.device("cpu")):
        self.T = T
        self.dt = dt
        self.y0 = 0.0
        self.g = 1.0
        self.a = a
        self.b = b
        self.n_bfs = w.shape[1]

        self.device = device

        # canonical system
        a = 1.0
        self.cs = CanonicalSystem(a, T, dt)

        # initialize basis functions for LWR
        self.w = w.to(self.device)
        self.centers = None
        self.widths = None
        self.set_basis_functions()

        # executed trajectory
        self.y = None
        self.yd = None
        self.z = None
        self.zd = None

        # desired path
        self.path = None

        self.reset()

    def reset(self):
        if self.y0 is not None:
            self.y = self.y0  # .copy()
        else:
            self.y0 = 0.0
            self.y = 0.0
        self.yd = 0.0
        self.z = 0.0
        self.zd = 0.0
        self.cs.reset()

    def set_basis_functions(self):
        time = torch.linspace(0, self.T, self.n_bfs).to(self.device)
        self.centers = torch.zeros(self.n_bfs).to(self.device)
        self.centers = torch.exp(-self.cs.a * time).to(self.device)
        self.widths = torch.ones(self.n_bfs).to(self.device) * self.n_bfs ** 1.5 / self.centers / self.cs.a

    def psi(self, theta):
        return torch.exp(-self.widths * (theta - self.centers) ** 2)

    def step(self, tau=1.0, k=1.0, start=None, goal=None):
        if goal is None:
            g = self.g
        else:
            g = goal

        if start is None:
            y0 = self.y0
        else:
            y0 = start

        theta = self.cs.step(tau)
        psi = self.psi(theta)

        f = torch.matmul(self.w, psi) * theta * k * (g - y0) / torch.sum(psi).item()

        self.zd = self.a * (self.b * (g - self.y) - self.z) + f  # transformation system
        self.zd /= tau

        self.z += self.zd * self.dt

        self.yd = self.z / tau
        self.y += self.yd * self.dt
        return self.y, self.yd, self.z, self.zd

    def fit(self, y_demo, tau=1.0):
        self.path = y_demo
        self.y0 = y_demo[0].copy()
        self.g = y_demo[-1].copy()

        y_demo = utils.interpolate_path(self, y_demo)
        yd_demo, ydd_demo = utils.calc_derivatives(y_demo, self.dt)

        f_target = tau**2 * ydd_demo - self.a * (self.b * (self.g - y_demo) - tau * yd_demo)
        f_target /= (self.g - self.y0)

        theta_seq = self.cs.all_steps().numpy()
        psi_funs = self.psi(theta_seq).numpy()

        # Locally Weighted Regression
        aa = np.multiply(theta_seq.reshape((1, theta_seq.shape[0])), psi_funs.T)
        aa = np.multiply(aa, f_target.reshape((1, theta_seq.shape[0])))
        aa = np.sum(aa, axis=1)

        bb = np.multiply(theta_seq.reshape((1, theta_seq.shape[0])) ** 2, psi_funs.T)
        bb = np.sum(bb, axis=1)
        self.w = torch.from_numpy(aa / bb)

        self.reset()

    def run_sequence(self, tau=1.0, k=1.0, start=None, goal=None):
        y = torch.zeros(self.w.shape[0],self.cs.N).to(self.device)
        y[0] = self.y0
        for i in range(self.cs.N):
            y[:,i], _, _, _ = self.step(tau=tau, k=k, start=start, goal=goal)
        return y
