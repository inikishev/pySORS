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

from ._minimize import minimize, ALL_METHODS
from .stp import STP
from .bds import BDS
from .ahds import AHDS
from .rs import RS
from .rspi import RSPI_FD, RSPI_SPSA

# remove modules from __all__
import types # pylint:disable=C0411
__all__ = [name for name, thing in globals().items() # type:ignore
          if not (name.startswith('_') or isinstance(thing, types.ModuleType))]
del types