import time
from collections.abc import Callable, Sequence
from typing import Literal, Optional, Any

import numpy as np

from .ahds import AHDS
from .bds import BDS
from .rs import RS
from .rspi import RSPI_FD, RSPI_SPSA
from .stp import STP
from .utils import SecondOrderRandomSearchOptimizer

__all__ = [
    "minimize",
    "ALL_METHODS"
]

ALL_METHODS: dict[Literal['stp', 'bds', 'ahds', 'rs', 'rspifd', 'rspispsa'], type[SecondOrderRandomSearchOptimizer]] = {
    "stp": STP,
    "bds": BDS,
    'ahds': AHDS,
    'rs': RS,
    'rspifd': RSPI_FD,
    'rspispsa': RSPI_SPSA
}

class EndMinimize(Exception): pass

class _Function:
    """Wraps the function and raises an EndMinimize exception when any stopping criteria is met.
    This is because all methods do multiple evaluations per step.
    """
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        maxfun: Optional[int],
        maxtime: Optional[float],
        stopval: Optional[float],
        max_no_improve: Optional[int],
        callbacks: Sequence[Callable[[np.ndarray, float], Any]],
    ):
        self.f = f
        self.maxfun = maxfun
        self.maxtime = maxtime
        self.stopval = stopval
        self.max_no_improve = max_no_improve
        self.callbacks = callbacks

        self.start_time = time.time()

        self.nfun = 0

        self.lowest_value = np.inf
        self.x = np.empty(0)
        self.no_improve_evals = 0

    def __call__(self, x: np.ndarray):
        value = self.f(x)

        if value < self.lowest_value:
            self.lowest_value = value
            self.x = x
            self.no_improve_evals = 0
        else:
            self.no_improve_evals += 1
            if self.max_no_improve is not None and self.no_improve_evals >= self.max_no_improve: raise EndMinimize()

        # stop conditions
        if self.maxfun is not None and self.nfun >= self.maxfun: raise EndMinimize()
        if self.maxtime is not None and time.time() - self.start_time >= self.maxtime: raise EndMinimize()
        if self.stopval is not None and value <= self.stopval: raise EndMinimize()
        if self.max_no_improve is not None and self.no_improve_evals >= self.max_no_improve: raise EndMinimize()

        for cb in self.callbacks: cb(x, value)
        self.nfun += 1
        return value



class Result:
    """Result of optimization. Important attributes are `x` - solution array, and `value` - value of objective function at `x`."""
    def __init__(self, objective: _Function, niter: int):
        self.time_passed = time.time() - objective.start_time
        """Time passed since start of optimization"""
        self.x = objective.x
        """Solution array."""
        self.nfun = objective.nfun
        """Number of function evaluations."""
        self.niter = niter
        """Number of optimizer iterations."""
        self.value = objective.lowest_value
        """Lowest value, which is achived under `x`."""

    def __repr__(self):
        return f"lowest value: {self.value}\nnumber of function evaluations: {self.nfun}\nYou can access the solution array under `x` attribute."

def minimize(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray | Sequence,
    method: Literal['stp', 'bds', 'ahds', 'rs', 'rspifd', 'rspispsa'] | str | SecondOrderRandomSearchOptimizer,
    maxfun: Optional[int] = None,
    maxiter: Optional[int] = None,
    maxtime: Optional[float] = 60,
    stopval: Optional[float] = None,
    max_no_improve: Optional[int] = 1000,
    allow_no_stop = False,
    callbacks: Optional[Callable[[np.ndarray, float], Any] | Sequence[Callable[[np.ndarray, float], Any]]] = None
    ):
    """Minimization of scalar function of one or more variables.

    :param f: The objective function to be minimized. It should accept a single argument, a numpy.ndarray of same shape as `x0`, and return a scalar value.
    :param x0: Initial argument for `f` to start from. You can use np.random.uniform to generate one.
    :param method: The optimization method, either a string or an instance of SecondOrderRandomSearchOptimizer. Please note that those optimizers have a lot of hyperparameters, and accessing them by string will use the default hyperparameters that might not be optimal for your problem.
    :param maxfun: Maximum number of function evaluations. defaults to None
    :param maxiter: Maximum number of optimizer steps, optimizers do multiple evaluations per step. defaults to None
    :param maxtime: Maximum time to run minimization for, in seconds. Defaults to 60
    :param stopval: Stop optimization once objective value is less or equal to this. defaults to None
    :param max_no_improve: Stops after this many consecutive evaluations of objective function with no improvement in the objective value, defaults to 1000
    :param allow_no_stop: If True, won't raise an exception when no stopping condition is specified. Note that this means the minimization will run forever, unless you manually stop it. Defaults to False
    :param callbacks: Function or sequence of functions that take in array and objective value at that array: `callback(x: np.ndarray, f(x): float)`.
    :return: Result of optimization. Important attributes are `x` - solution array, and `value` - value of objective function at `x`.


    example:
    ```py
    import pysors
    import numpy as np

    def rosenbrock(arr):
        x,y = arr
        a = 1
        b = 100
        return (a - x) ** 2 + b * (y - x ** 2) ** 2

    x0 = np.array([-3., -4.])
    res = pysors.minimize(rosenbrock, x0 = x0, method = 'bds', stopval=1e-8)

    print(res) # - optimization result, holds `x`, `value` attributes
    print(res.x) # - solution array.
    ```
    """
    # check that there is a stopping condition
    if (not allow_no_stop) and all(i is None for i in [maxfun, maxiter, maxtime, stopval, max_no_improve]):
        raise ValueError('All stopping conditions are disabled, this will run forever. '
                         'Please set one of [maxfun, maxiter, maxtime, stopval, max_no_improve], '
                         'or set `allow_no_stop` to True if you intend to stop the function manually')
    if callbacks is None: callbacks = ()
    if callable(callbacks): callbacks = (callbacks, )

    # get the method
    if isinstance(method, str):
        norm_str = ''.join([char for char in method.lower() if char.isalpha()])
        if norm_str in ALL_METHODS: optimizer = ALL_METHODS[norm_str]() # type:ignore
        else: raise KeyError(f'Method "{method}" is not a valid method. Valid methods methods are {tuple(ALL_METHODS.keys())}')
    else: optimizer = method


    # optimize
    objective = _Function(f, maxfun=maxfun, maxtime=maxtime, stopval=stopval, max_no_improve=max_no_improve, callbacks = callbacks)
    x = np.array(x0, copy = False)
    cur_iter = 0
    while True:
        try:
            x = optimizer.step(objective, x)
        except EndMinimize:
            break

        cur_iter += 1
        if maxiter is not None and cur_iter >= maxiter: break


    return Result(objective, cur_iter)