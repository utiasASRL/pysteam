import numpy as np

from .state_var import StateVar


class VectorSpaceStateVar(StateVar):
  """Variable wrapper of a nx1 numpy array."""

  def __init__(self, value: np.ndarray, **kwargs) -> None:
    assert len(value.shape) == 2 and value.shape[-1] == 1
    super().__init__(value, value.shape[-2], **kwargs)

  def get_value(self) -> np.ndarray:
    return self._value

  def set_value(self, value: np.ndarray) -> None:
    self._value[:] = value

  def update(self, perturbation: np.ndarray) -> None:
    self._value += perturbation

  def clone(self):
    raise NotImplementedError