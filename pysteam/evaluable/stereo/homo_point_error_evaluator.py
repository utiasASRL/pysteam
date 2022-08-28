import numpy as np

from ..evaluable import Evaluable, Node, Jacobians


class HomoPointErrorEvaluator(Evaluable):
  """Currently just ignore the last entry."""

  def __init__(self, pt: Evaluable, meas_pt: np.ndarray):
    super().__init__()
    self._pt: Evaluable = pt
    self._meas_pt: np.ndarray = meas_pt
    self._D = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])

  @property
  def active(self) -> bool:
    return self._pt.active

  @property
  def related_var_keys(self) -> set:
    return self._pt.related_var_keys

  def forward(self) -> Node:
    child = self._pt.forward()
    value = self._D @ (self._meas_pt - child.value)
    return Node(value, child)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._pt.active:
      self._pt.backward(-lhs @ self._D, node.children[0], jacs)
