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


class RS(SecondOrderRandomSearchOptimizer):
    """Two-step random search (RS)."""
    def __init__(
        self,
        a_init=0.25,
        sigma_1=0.5,
        sigma_2=0.25,
        distribution:Literal['Uniform', 'Normal']='Normal',
        step_upd: Literal['half', 'inv', 'inv_sqrt']="half",
        theta=0.6,
        T_half=10,
    ):
        """Two-step random search.
        '''
        Similar to the STP method [6], but we alternate
        between two perturbation magnitudes: sigma1 is set to be optimal for the large gradient case, while sigma2 optimal to
        escape saddles.
        '''

        :param a_init: Initial step size, defaults to 0.25
        :param sigma_1: First sigma, defaults to 0.5
        :param sigma_2: Second sigma, defaults to 0.25
        :param distribution: Random petrubation distribution, "Normal" or "Uniform", defaults to "Normal"
        :param step_upd: Step update rule, defaults to "half"
        :param theta: Multiplier to step size on every `T_half` steps if `step_upd = "half"`, defaults to 0.6
        :param T_half: Multiply step size by `theta` every `T_half` steps if `step_upd = "half"`, defaults to 10
        """
        super().__init__()
        self.a = a_init
        self.a_init = a_init
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.step_upd = step_upd
        self.distribution = distribution
        self.theta = theta
        self.T_half = T_half

        self.t = 1

    def step(self, f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        self._initialize(f, x)

        # Initialization
        y = x.flatten() # iterate @ t

        #""" ========= Random Step 1 ========= """
        if self.distribution == 'Uniform':
            d1 = np.random.multivariate_normal(np.zeros(self.d), np.identity(self.d))
            d1 = self.sigma_1*(d1/np.linalg.norm(d1))
        elif self.distribution == 'Normal':
            d1 = np.random.multivariate_normal(np.zeros(self.d), np.power(self.sigma_1,2.0)*np.identity(self.d))
        else:
            raise ValueError(f'The option {self.distribution} is not a supported sampling distribution.')

        V = [y, y+self.a*d1, y-self.a*d1]

        f_v = []
        for v in V:
            f_v.append(self.eval(v))

        # Select optimal point
        i_star = np.argmin(np.array(f_v))

        # Update iterate
        y = V[i_star]

        #""" ========= Random Step 2 ========= """
        if i_star == 0:
            if self.distribution == 'Uniform':
                d2 = np.random.multivariate_normal(np.zeros(self.d), np.identity(self.d))
                d2 = self.sigma_2*(d2/np.linalg.norm(d2))
            elif self.distribution == 'Normal':
                d2 = np.random.multivariate_normal(np.zeros(self.d), np.power(self.sigma_2,2.0)*np.identity(self.d))
            else:
                raise ValueError(f'The option {self.distribution} is not a supported sampling distribution.')

            V = [y, y+self.a*d2, y-self.a*d2]

            f_v = []
            for v in V:
                f_v.append(self.eval(v))

            # Select optimal point
            i_star = np.argmin(np.array(f_v))

            # Update iterate
            y = V[i_star]

        # Update step-size
        if self.step_upd == 'half':
            if self.t%self.T_half == 0:
                self.a = self.theta*self.a
        elif self.step_upd == 'inv':
            self.a = self.a_init/(self.t+1)
        elif self.step_upd == 'inv_sqrt':
            self.a = self.a_init/np.sqrt(self.t+1)
        else:
            raise ValueError(f'The option {self.step_upd} is not a supported step size update rule.')


        self.t += 1
        return y.reshape(x.shape)