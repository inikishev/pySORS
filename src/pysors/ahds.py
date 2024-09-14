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

import itertools
from collections.abc import Callable

import numpy as np

from .utils import SecondOrderRandomSearchOptimizer, _rho, uniform_angles_pss


class AHDS(SecondOrderRandomSearchOptimizer):
    """Approximate Hessian Direct Search"""
    def __init__(self, a_init = 0.25, a_max = 10., theta  = 0.9, gamma = 1.1, rho = _rho):
        """Approximate Hessian Direct Search.

        :param a_init: Initial step size, defaults to 0.25
        :param a_max: Max step size, defaults to 10.
        :param theta: Multiplier to step size on each unsuccessful step, defaults to 0.9
        :param gamma: Multiplier to step size on each successful step, defaults to 1.1
        :param rho: Defaults to _rho (not sure what is is)
        """
        super().__init__()
        self.a = a_init
        self.a_init = a_init
        self.a_max = a_max
        self.theta = theta
        self.gamma = gamma
        self.rho = rho

        self.stacked = False

        self.t = 1

    def step(self, f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray: 
        self._initialize(f, x)

        # Initialization
        y = x.flatten() # iterate @ t

        # Reset variables
        successful = False
        H = np.zeros((self.d, self.d)) # Hessian
        f_y = self.eval(y) # function value at current iterate
        d_opt = np.zeros(self.d) # descent direction
        B_opt = np.zeros((self.d, self.d)) # independent set of vectors
        D_table = np.zeros(self.d) # store function values for Hessian computation
        B_table = np.zeros((self.d, self.d)) # store function values for Hessian computation

        # """ ========= Step 1 ========= """
        # Generate a PSS D
        D, D_symmetric, D_size = uniform_angles_pss(self.d)

        # Search the PSS
        for i in np.random.permutation(D_size):
            d = D[:,i]
            if self.eval(y + self.a*d) < f_y - self.rho(self.a):
                # Iteration succesful
                d_opt = d
                successful = True
                # Stop searching PSS
                break

        # """ ========= Step 2 ========= """
        # Search opposite directions in PSS
        if not D_symmetric and not successful:
            for i in np.random.permutation(D_size):
                d = -D[:,i]
                if self.eval(y + self.a*d) < f_y - self.rho(self.a):
                    # Iteration succesful
                    d_opt = d
                    successful = True
                    # Stop searching PSS
                    break

        # """ ========= Step 3 ========= """
        if not successful:
            # Choose B as a subset of D with f.d linearly independent vectors
            subsets = itertools.combinations(range(D_size), self.d)
            for subset in np.random.permutation(list(subsets)):
                B = D[:,subset]
                if np.linalg.matrix_rank(B) == self.d:
                    B_opt = B
                    # Stop search
                    break

            break_outer = False
            for i in range(self.d-1):
                for j in range(i+1,self.d):
                    d = B_opt[:,i] + B_opt[:,j]
                    B_table[i,j] = self.eval(y + self.a*d)
                    if B_table[i,j] < f_y - self.rho(self.a):
                        # Iteration successful
                        d_opt = d
                        successful = True
                        # Stop searching
                        break_outer = True
                        break
                if break_outer:
                    # Stop searching
                    break

        # """ ========= Step 4 ========= """
        # if not successful and not stacked:
        if not successful:
            # Hessian approximation: Diagonal elements
            for i in range(self.d):
                di = B_opt[:,i]
                D_table[i] = self.eval(y+self.a*di)
                H[i,i] = D_table[i] - 2*f_y + self.eval(y-self.a*di)

            # Hessian approximation: Off-diagonal elements
            for i in range(self.d-1):
                for j in range(i+1,self.d):
                    H[i,j] = B_table[i,j] - D_table[i] - D_table[j] + f_y
                    H[j,i] = H[i,j]

            # Complete computation
            H = H/(self.a**2)

            # When iterates get very close to a minimizer the Hessian approximation
            # may result to NaN values. The try statement avoids such errors.
            try:
                # Eigendecomposition
                L, V = np.linalg.eig(H)

                # Eigenvector corresponding to minimum eigenvalue
                idx = np.argmin(L)
                d = V[:,idx]

                # Check d
                if self.eval(y + self.a*d) < self.eval(y) - self.rho(self.a):
                    # Iteration successful
                    d_opt = d
                    successful = True

                # Check -d
                if self.eval(y - self.a*d) < self.eval(y) - self.rho(self.a) and self.eval(y - self.a*d) < self.eval(y + self.a*d):
                    # Iteration successful
                    d_opt = -d
                    successful = True
            except Exception:
                pass

        # """ ========= Step 5 ========= """
        # Update step
        if successful:
            y = y + self.a*d_opt
            self.a = np.minimum(self.gamma*self.a, self.a_max)
            self.stacked = False
        else:
            self.a = self.theta*self.a
            self.stacked = True

        self.t += 1
        return y.reshape(x.shape)