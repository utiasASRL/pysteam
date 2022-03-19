from typing import Dict
import numpy as np

from pylgmath import se3op

from ..state_key import StateKey
from ..evaluable import Evaluable, Node


class P2PErrorEvaluator(Evaluable):
  """The distance between two points living in their respective frame."""

  def __init__(
      self,
      T_rq: Evaluable,
      reference: np.ndarray,
      query: np.ndarray,
  ) -> None:
    super().__init__()

    self._T_rq: Evaluable = T_rq
    self._reference: np.ndarray = reference / reference[3]
    self._query: np.ndarray = query / query[3]

    self._D = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])

  @property
  def active(self) -> bool:
    return self._T_rq.active

  def forward(self) -> Node:
    child = self._T_rq.forward()
    value = self._D @ (self._reference - child.value.matrix() @ self._query)
    return Node(value, child)

  def backward(self, lhs, node) -> Dict[StateKey, np.ndarray]:
    jacs = dict()
    child = node.children[0]
    if self._T_rq.active:
      T_rq = child.value.matrix()
      lhs = -lhs @ self._D @ se3op.point2fs(T_rq @ self._query)
      jacs = self._T_rq.backward(lhs, child)
    return jacs