"""
Code from https://github.com/adamsolomou/second-order-random-search

@inproceedings{
  lucchi2021randomsearch,
  title={On the Second-order Convergence Properties of Random Search Methods},
  author={Aurelien Lucchi and Antonio Orvieto and Adamos Solomou},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}

LICENSE: Apache 2.0
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np


def uniform_angles_pss(dim): 
	# Define A
	A = np.empty((dim,dim))
	for i in range(dim): 
		for j in range(dim): 
			if i==j: 
				A[i,j] = 1
			else:
				A[i,j] = - 1/dim

	# Find C such that A = CC^T
	C = np.linalg.cholesky(A)

	# V = C^T
	V = np.transpose(C)

	v = -np.sum(V, axis=1)
	v = np.reshape(v, (dim,1))

	# Minimal PSS
	D = np.hstack((V,v))

	# Size of PSS
	size = D.shape[1]

	# Flag indicating if PSS is symmetric or not 
	symmetric = False

	return D, symmetric, size 

def _rho(z, p=2, c=0.0): return c*z**p


class SecondOrderRandomSearchOptimizer(ABC):
    f: Callable[[np.ndarray], float]
    shape: Sequence[int]
    d: int

    def eval(self, x: np.ndarray) -> float:
        return self.f(x.reshape(self.shape))

    def _initialize(self, f: Callable[[np.ndarray], float], x: np.ndarray):
        self.f = f
        self.shape = x.shape
        self.d = x.size

    @abstractmethod
    def step(self, f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray: ...