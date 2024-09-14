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

from .utils import SecondOrderRandomSearchOptimizer, _rho, uniform_angles_pss


class BDS(SecondOrderRandomSearchOptimizer):
    """Basic Direct Search"""
    def __init__(self, a_init = 0.25, a_max = 10., theta  = 0.9, gamma = 1.1, rho = _rho):
        """Basic Direct Search.

        :param a_init: Initial step size, defaults to 0.25
        :param a_max: Maximum step size, defaults to 10.
        :param theta: Multiplier to step size on each unsuccessful step, defaults to 0.9
        :param gamma: Multiplier to step size on each successful step, defaults to 1.1
        :param rho: defaults to _rho (not sure what is is)
        """
        super().__init__()
        self.a = a_init
        self.a_init = a_init
        self.a_max = a_max
        self.theta = theta
        self.gamma = gamma
        self.rho = rho

        self.t = 1

    def step(self, f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray: 
        self._initialize(f, x)

        # Initialization
        y = x.flatten() # iterate @ t

        # Reset variables
        successful = False
        d_opt = np.zeros(self.d)
        f_y = self.eval(y) # function value at current iterate

        # Generate a polling set
        D, D_symmetric, D_size = uniform_angles_pss(self.d)

        # Search the polling set
        for i in np.random.permutation(D_size):
            d = D[:,i]
            if self.eval(y + self.a*d) < f_y - self.rho(self.a):
                # Iteration succesful
                d_opt = d
                successful = True
                # Stop searching PSS
                break

        # Update step
        if successful:
            y = y + self.a*d_opt
            self.a = np.minimum(self.gamma*self.a, self.a_max)
        else:
            self.a = self.theta*self.a

        self.t += 1
        return y.reshape(x.shape)