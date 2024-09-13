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


def _DFPI_SPSA(f, y, c_init, beta, T_power):
	# Power iteration - Compute eigenvector for max eigenvalue
	r = 0.001
	T_power_approx = 5
	d2 = np.random.rand(f.d)

	c = c_init
	for i in range(T_power_approx):
		Delta = np.random.binomial(n=1, p=0.5, size=f.d)
		Delta[Delta == 0] = -1
		# Approximate gradient vectors
		d_rplus = f.eval(y + r*d2 + c*Delta) - f.eval(y + r*d2 - c*Delta)
		G_rplus = np.divide(d_rplus, 2*c*Delta)

		d_rminus = f.eval(y - r*d2 + c*Delta) - f.eval(y - r*d2 - c*Delta)
		G_rminus = np.divide(d_rminus, 2*c*Delta)

		# Approximate Hessian-vector product
		Hd = (G_rplus - G_rminus)/(2*r)

		# Power iteration - update
		d2 = Hd/np.linalg.norm(Hd)

	# Approximate gradient vectors
	d_rplus = f.eval(y + r*d2 + c*Delta) - f.eval(y + r*d2 - c*Delta) # type:ignore
	G_rplus = np.divide(d_rplus, 2*c*Delta)# type:ignore

	d_rminus = f.eval(y - r*d2 + c*Delta) - f.eval(y - r*d2 - c*Delta)# type:ignore
	G_rminus = np.divide(d_rminus, 2*c*Delta)# type:ignore

	# Approximate Hessian-vector product
	Hd = (G_rplus - G_rminus)/(2*r)

	# Largest eigenvalue
	lmax = np.linalg.norm(Hd)/np.linalg.norm(d2)

	# Power iteration - Compute eigenvector for min eigenvalue
	b_power = 1/lmax
	d2 = np.random.rand(f.d)
	for i in range(T_power):
		Delta = np.random.binomial(n=1, p=0.5, size=f.d)
		Delta[Delta == 0] = -1

		# Approximate gradient vectors
		d_rplus = f.eval(y + r*d2 + c*Delta) - f.eval(y + r*d2 - c*Delta)
		G_rplus = np.divide(d_rplus, 2*c*Delta)

		d_rminus = f.eval(y - r*d2 + c*Delta) - f.eval(y - r*d2 - c*Delta)
		G_rminus = np.divide(d_rminus, 2*c*Delta)

		# Approximate Hessian-vector product
		Hd = (G_rplus - G_rminus)/(2*r)

		# Power iteration - update
		d2_ = d2 - b_power*Hd
		d2  = d2_/np.linalg.norm(d2_)

	# Negative curvature
	return d2

def _DFPI_FD(f, y, c, T_power):
	r = 0.01

	# Power iteration - Compute eigenvector for max eigenvalue
	T_power_approx = 15
	d2 = np.random.rand(f.d)

	# Basis vectors
	I = np.identity(f.d)

	for i in range(T_power_approx):
		# Initialize
		g_p = np.empty(f.d)
		g_m = np.empty(f.d)

		# Approximate gradient vectors
		for j in range(f.d):
			g_p[j] = (f.eval(y + r*d2 + c*I[:,j]) - f.eval(y + r*d2 - c*I[:,j]))/(2*c)
			g_m[j] = (f.eval(y - r*d2 + c*I[:,j]) - f.eval(y - r*d2 - c*I[:,j]))/(2*c)

		# Approximate Hessian-vector product
		Hd = (g_p - g_m)/(2*r)

		# Power iteration - update
		d2 = Hd/np.linalg.norm(Hd)

	# Approximate gradient vectors
	g_p = np.empty(f.d)
	g_m = np.empty(f.d)

	for j in range(f.d):
		g_p[j] = (f.eval(y + r*d2 + c*I[:,j]) - f.eval(y + r*d2 - c*I[:,j]))/(2*c)
		g_m[j] = (f.eval(y - r*d2 + c*I[:,j]) - f.eval(y - r*d2 - c*I[:,j]))/(2*c)

	# Approximate Hessian-vector product
	Hd = (g_p - g_m)/(2*r)

	# Largest eigenvalue
	lmax = np.linalg.norm(Hd)/np.linalg.norm(d2)

	# Power iteration - Compute eigenvector for min eigenvalue
	b_power = 1/lmax
	d2 = np.random.rand(f.d)

	for i in range(T_power):
		# Initialize
		g_p = np.empty(f.d)
		g_m = np.empty(f.d)

		# Approximate gradient vectors
		for j in range(f.d):
			g_p[j] = (f.eval(y + r*d2 + c*I[:,j]) - f.eval(y + r*d2 - c*I[:,j]))/(2*c)
			g_m[j] = (f.eval(y - r*d2 + c*I[:,j]) - f.eval(y - r*d2 - c*I[:,j]))/(2*c)

		# Approximate Hessian-vector product
		Hd = (g_p - g_m)/(2*r)

		# Power iteration - update
		d2_ = d2 - b_power*Hd
		d2  = d2_/np.linalg.norm(d2_)

	return d2



class RSPI_SPSA(SecondOrderRandomSearchOptimizer):
	"""Power Iteration Random Search with SPSA hessian estimation"""
	def __init__(
		self,
		a_init=0.25,
		c_init=0.1,
		beta=0.101,
		sigma_1=0.5,
		sigma_2=0.25,
		distribution="Normal",
		step_upd="half",
		theta=0.6,
		T_half=10,
		T_power=100,
	):
		super().__init__()
		self.a = a_init
		self.c = c_init
		self.a_init = a_init
		self.c_init = c_init
		self.beta = beta
		self.sigma_1 = sigma_1
		self.sigma_2 = sigma_2
		self.distribution = distribution
		self.step_upd = step_upd
		self.theta = theta
		self.T_half = T_half
		self.T_power = T_power

		self.t = 1

	def step(self, f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
		"""Perform one optimization step

		:param f: function that takes in and array of same shape as `x` and outputs a scalar value.
		:param x: Current parameters.
		:return: `x` new parameters.

		example:
		```py
		x = np.array([-3., -4.])
		opt = pysors.BDS()

		for i in range(1000):
			x = opt.step(rosenbrock, x)

		print(x) # last solution array
		print(rosenbrock(x)) # objective value at x
		```
		"""
		self._initialize(f, x)

		# Initialization
		y = x.flatten()  # iterate @ t

		# """ ========= Random Step ========= """
		if self.distribution == "Uniform":
			d1 = np.random.multivariate_normal(np.zeros(self.d), np.identity(self.d))
			d1 = self.sigma_1 * (d1 / np.linalg.norm(d1))
		elif self.distribution == "Normal":
			d1 = np.random.multivariate_normal(
				np.zeros(self.d), np.power(self.sigma_1, 2.0) * np.identity(self.d)
			)
		else:
			raise ValueError(
				f"The option {self.distribution} is not a supported sampling distribution."
			)

		V = [y, y + self.a * d1, y - self.a * d1]

		f_v = []
		for v in V:
			f_v.append(self.eval(v))

		# Select optimal point
		i_star = np.argmin(np.array(f_v))

		# Update iterate
		y = V[i_star]

		# """ ========= Negative Curvature ========= """
		if i_star == 0:
			d2 = _DFPI_SPSA(self, y, self.c, self.beta, self.T_power)

			while d2 is None:
				d2 = _DFPI_SPSA(self, y, self.c, self.beta, self.T_power)

			# """ ========= Update Step ========= """
			V = [y, y + self.sigma_2 * d2, y - self.sigma_2 * d2]

			f_v = []
			for v in V:
				f_v.append(self.eval(v))

			# Select optimal point
			i_star = np.argmin(np.array(f_v))

			# Update iterate
			y = V[i_star]

		# Decrease SPSA parameter
		self.c = self.c_init / pow(self.t, self.beta)

		# Update step-size
		if self.step_upd == "half":
			if self.t % self.T_half == 0:
				self.a = self.theta * self.a
		elif self.step_upd == "inv":
			self.a = self.a_init / (self.t + 1)
		elif self.step_upd == "inv_sqrt":
			self.a = self.a_init / np.sqrt(self.t + 1)
		else:
			raise ValueError(
				f"The option {self.step_upd} is not a supported step size update rule."
			)

		self.t += 1
		return y.reshape(x.shape)


class RSPI_FD(SecondOrderRandomSearchOptimizer):
	"""Power Iteration Random Search"""
	def __init__(
		self,
		a_init=0.25,
		c_init=0.1,
		beta=0.101,
		sigma_1=0.5,
		sigma_2=0.25,
		distribution="Normal",
		step_upd="half",
		theta=0.6,
		T_half=10,
		T_power=100,
	):
		super().__init__()
		self.a = a_init
		self.c = c_init
		self.a_init = a_init
		self.c_init = c_init
		self.beta = beta
		self.sigma_1 = sigma_1
		self.sigma_2 = sigma_2
		self.distribution = distribution
		self.step_upd = step_upd
		self.theta = theta
		self.T_half = T_half
		self.T_power = T_power

		self.t = 1

	def step(self, f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
		self._initialize(f, x)

		# Initialization
		y = x.flatten() # iterate @ t


		# """========= Random Step ========="""
		if self.distribution == "Uniform":
			d1 = np.random.multivariate_normal(np.zeros(self.d), np.identity(self.d))
			d1 = self.sigma_1 * (d1 / np.linalg.norm(d1))
		elif self.distribution == "Normal":
			d1 = np.random.multivariate_normal(
				np.zeros(self.d), np.power(self.sigma_1, 2.0) * np.identity(self.d)
			)
		else:
			raise ValueError(f"The option {self.distribution} is not a supported sampling distribution.")

		V = [y, y + self.a * d1, y - self.a * d1]

		f_v = []
		for v in V:
			f_v.append(self.eval(v))

		# Select optimal point
		i_star = np.argmin(np.array(f_v))

		# Update iterate
		y = V[i_star]

		# """ ========= Negative Curvature ========= """
		if i_star == 0:
			d2 = _DFPI_FD(self, y, self.c, self.T_power)

			while d2 is None:
				d2 = _DFPI_FD(self, y, self.c, self.T_power)

			# """ ========= Update Step ========= """
			V = [y, y + self.sigma_2 * d2, y - self.sigma_2 * d2]

			f_v = []
			for v in V:
				f_v.append(self.eval(v))

			# Select optimal point
			i_star = np.argmin(np.array(f_v))

			# Update iterate
			y = V[i_star]

		# Decrease SPSA parameter
		self.c = self.c_init / pow(self.t,self.beta)

		# Update step-size
		if self.step_upd == "half":
			if self.t % self.T_half == 0:
				self.a = self.theta * self.a
		elif self.step_upd == "inv":
			self.a = self.a_init / (self.t + 1)
		elif self.step_upd == "inv_sqrt":
			self.a = self.a_init / np.sqrt(self.t + 1)
		else:
			raise ValueError(f"The option {self.step_upd} is not a supported step size update rule.")

		self.t += 1
		return y.reshape(x.shape)
