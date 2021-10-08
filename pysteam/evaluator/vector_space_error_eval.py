from typing import Optional
import numpy as np

from . import Evaluator
from ..state import VectorSpaceStateVar


class VectorSpaceErrorEval(Evaluator):
  """Error evaluator for a measured vector space state variable"""

  def __init__(self, meas: np.ndarray, state_vec: VectorSpaceStateVar) -> None:
    super().__init__()
    self._meas: np.ndarray = meas
    self._state_vec: VectorSpaceStateVar = state_vec

  def is_active(self):
    return not self._state_vec.is_locked()

  def evaluate(self, lhs: Optional[np.ndarray] = None):
    error = self._meas - self._state_vec.get_value()

    if lhs is None:
      return error

    assert lhs.shape[-1] == self._state_vec.get_perturb_dim()

    jacs = dict()

    if not self._state_vec.is_locked():
      jacs = {self._state_vec.get_key(): -lhs}

    return error, jacs