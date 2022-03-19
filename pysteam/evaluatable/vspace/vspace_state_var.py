import numpy as np

from ..state_var import StateVar
from ..evaluatable import Node


class VSpaceStateVar(StateVar):
  """Variable wrapper of a nx1 numpy array."""

  def __init__(self, value: np.ndarray, **kwargs) -> None:
    assert len(value.shape) == 2 and value.shape[-1] == 1
    super().__init__(value.shape[-2], **kwargs)
    self._value = value

  def forward(self):
    return Node(self._value)

  def backward(self, lhs, node):
    return {self.key: lhs} if self.active else {}

  def clone(self):
    raise NotImplementedError

  @property
  def value(self) -> np.ndarray:
    return self._value

  def assign(self, value: np.ndarray) -> None:
    self._value[:] = value

  def update(self, perturbation: np.ndarray) -> None:
    self._value += perturbation
