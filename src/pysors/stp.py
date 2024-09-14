"""
Code modified from https://github.com/adamsolomou/second-order-random-search

@inproceedings{
  lucchi2021randomsearch,
  title={On the Second-order Convergence Properties of Random Search Methods},
  author={Aurelien Lucchi and Antonio Orvieto and Adamos Solomou},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}

LICENSE: Apache 2.0
"""

from collections.abc import Callable
from typing import Literal

import numpy as np

from .utils import SecondOrderRandomSearchOptimizer


class STP(SecondOrderRandomSearchOptimizer):
    """Stochastic Three Points"""

    def __init__(
        self,
        a_init=0.25,
        step_upd: Literal['half', 'inv', 'inv_sqrt']="half",
        distribution: Literal["Uniform", "Normal"] = "Uniform",
        theta: float = 0.5,
        T_half=10,
    ):
        """Stochastic Three Points.

        :param a_init: Initial step size, defaults to 0.25
        :param step_upd: Step update rule, defaults to "half"
        :param distribution: Random petrubation distribution, "Normal" or "Uniform", defaults to "Normal"
        :param theta: Multiplier to step size on every `T_half` steps if `step_upd = "half"`, defaults to 0.5
        :param T_half: Multiply step size by `theta` every `T_half` steps if `step_upd = "half"`, defaults to 10
        """
        super().__init__()
        self.a = a_init
        self.a_init = a_init
        self.step_upd = step_upd
        self.distribution = distribution
        self.theta = theta
        self.T_half = T_half

        self.t = 1

    def step(self, f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        self._initialize(f, x)

        # Initialization
        y = x.flatten()

        if self.distribution == 'Uniform':
            s = np.random.multivariate_normal(np.zeros(self.d), np.identity(self.d))
            s = s/np.linalg.norm(s)
        elif self.distribution == 'Normal':
            s = np.random.multivariate_normal(np.zeros(self.d), np.identity(self.d))
        else:
            raise ValueError(f'The option {self.distribution} is not a supported sampling distribution.')

        # List possible next iterates
        V = [y+self.a*s, y-self.a*s, y]

        f_v = []
        for v in V:
            f_v.append(self.eval(v))

        # Select optimal point
        i_star = np.argmin(np.array(f_v))

        # Update step
        y = V[i_star]

        # Step size update
        if self.step_upd == 'half':
            if self.t%self.T_half == 0:
                self.a = self.a * self.theta
        elif self.step_upd == 'inv':
            self.a = self.a_init/(self.t+1)
        elif self.step_upd == 'inv_sqrt':
            self.a = self.a_init/np.sqrt(self.t+1)
        else:
            raise ValueError(f'The option {self.step_upd} is not a supported step size update rule.')

        self.t += 1
        return y.reshape(x.shape)