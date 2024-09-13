import time
from collections.abc import Callable, Sequence
from typing import Literal, Optional

import numpy as np

from .ahds import AHDS
from .bds import BDS
from .rs import RS
from .rspi import RSPI_FD, RSPI_SPSA
from .stp import STP
from .utils import SecondOrderRandomSearchOptimizer

ALL_METHODS: dict[Literal['stp', 'bds', 'ahds', 'rs', 'rspifd', 'rspispsa'], type[SecondOrderRandomSearchOptimizer]] = {
    "stp": STP,
    "bds": BDS,
    'ahds': AHDS,
    'rs': RS,
    'rspifd': RSPI_FD,
    'rspispsa': RSPI_SPSA
}

class EndMinimize(Exception): pass

class Function:
    """Wraps the function and raises an EndMinimize exception when any stopping criteria is met.
    This is because all methods do multiple evaluations per step.
    """
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        maxfun: Optional[int] = None,
        maxtime: Optional[float] = 60,
        stopval: Optional[float] = None,
        max_no_improve: Optional[int] = 100,
    ):
        self.f = f
        self.maxfun = maxfun
        self.maxtime = maxtime
        self.stopval = stopval
        self.max_no_improve = max_no_improve

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

        self.nfun += 1
        return value



class Result:
    def __init__(self, objective: Function, niter: int):
        self.time_passed = time.time() - objective.start_time
        self.x = objective.x
        self.nfun = objective.nfun
        self.niter = niter
        self.value = objective.lowest_value

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
    allow_no_stop = False
    ):
    # check that there is a stopping condition
    if (not allow_no_stop) and all(i is None for i in [maxfun, maxiter, maxtime, stopval, max_no_improve]):
        raise ValueError('All stopping conditions are disabled, this will run forever. '
                         'Please set one of [maxfun, maxiter, maxtime, stopval, max_no_improve], '
                         'or set `allow_no_stop` to True if you intend to stop the function manually')

    # get the method
    if isinstance(method, str):
        norm_str = ''.join([char for char in method.lower() if char.isalpha()])
        if norm_str in ALL_METHODS: optimizer = ALL_METHODS[norm_str]() # type:ignore
        else: raise KeyError(f'Method "{method}" is not a valid method. Valid methods methods are {tuple(ALL_METHODS.keys())}')
    else: optimizer = method

    # optimize
    objective = Function(f, maxfun=maxfun, maxtime=maxtime, stopval=stopval, max_no_improve=max_no_improve)
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