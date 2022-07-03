import numpy as np
import numpy.linalg as npla

from ..state_var import StateVar
from ..evaluable import Node, Jacobians


class HomoPointStateVar(StateVar):
  """Variable wrapper of a 4x1 homogeneous point."""

  def __init__(self, value: np.ndarray, *, scale: bool = False, **kwargs) -> None:
    super().__init__(3, **kwargs)
    assert value.shape == (4, 1)
    if scale == False:
      assert value[3, 0] == 1.0
    self._value = value
    self._scale = scale
    self.refresh_homo_scaling()

  def forward(self):
    return Node(self._value)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self.active:
      jacs.add(self.key, lhs)

  @property
  def value(self) -> np.ndarray:
    return self._value

  def assign(self, value: np.ndarray) -> None:
    self._value[:] = value
    self.refresh_homo_scaling()

  def update(self, perturbation: np.ndarray) -> None:
    self._value[:3] += perturbation
    self.refresh_homo_scaling()

  def refresh_homo_scaling(self) -> None:
    if self._scale:
      # get length of xyz-portion of landmark
      mag = npla.norm(self._value[:3])
      if mag == 0.0:
        self._value[3] = 1.0
      else:
        # update to be unit length
        self._value /= mag
