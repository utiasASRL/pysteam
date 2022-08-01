import numpy as np

from pylgmath import se3op

from ..evaluable import Evaluable, Node, Jacobians


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
    if (reference.shape[0] == 3):
      self._reference: np.ndarray = np.ones((4, 1))
      self._reference[:3] = reference
      self._query: np.ndarray = np.ones((4, 1))
      self._query[:3] = query
    else:
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

  @property
  def related_var_keys(self) -> set:
    return self._T_rq.related_var_keys

  def forward(self) -> Node:
    child = self._T_rq.forward()
    value = self._D @ (self._reference - child.value.matrix() @ self._query)
    return Node(value, child)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._T_rq.active:
      child = node.children[0]
      T_rq = child.value.matrix()
      lhs = -lhs @ self._D @ se3op.point2fs(T_rq @ self._query)
      self._T_rq.backward(lhs, child, jacs)