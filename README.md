# pySORS

Fork of https://github.com/adamsolomou/second-order-random-search, which implements algorithms described in:
> Aurelien Lucchi, Antonio Orvieto, Adamos Solomou. [On the Second-order Convergence Properties of Random Search Methods](https://arxiv.org/abs/2110.13265). In Neural Information Processing Systems (NeurIPS), 2021.

This fork implements a scipy.minimize-like interface for those methods.

# Installation
https://pypi.org/project/pysors/
```
pip install pysors
```
If you are using conda, the only dependency this has is numpy, so you can safely install this with pip without messing up your conda environment.

# Usage
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

This can also be used step-wise in the following way:
```py
opt = pysors.BDS()
for i in range(1000):
    x = opt.step(rosenbrock, x)

print(x) # last solution array
print(rosenbrock(x)) # objective value at x
```

# List of methods
- `STP`: Stochastic Three Points
- `BDS`: Basic Direct Search
- `AHDS`: Approximate Hessian Direct Search
- `RS`: Two-step random search
- `RSPI_FD`: Power Iteration Random Search
- `RSPI_SPSA`: Power Iteration Random Search with SPSA hessian estimation

## References 

If you found this useful, please consider citing author's paper: 
```
@inproceedings{
  lucchi2021randomsearch,
  title={On the Second-order Convergence Properties of Random Search Methods},
  author={Aurelien Lucchi and Antonio Orvieto and Adamos Solomou},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```
