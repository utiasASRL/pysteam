from typing import Optional
import numpy as np

from pylgmath import se3op
from . import EvalTreeNode, Evaluator, TransformEvaluator


class PointToPointErrorEval(Evaluator):
  """The distance between two points living in their respective frame."""

  def __init__(
      self,
      T_rq: TransformEvaluator,
      reference: np.ndarray,
      query: np.ndarray,
  ) -> None:
    super().__init__()

    self._T_rq: TransformEvaluator = T_rq
    self._reference: np.ndarray = reference / reference[3]
    self._query: np.ndarray = query / query[3]

    self._D = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])

  def is_active(self):
    return self._T_rq.is_active()

  def evaluate(self, lhs: Optional[np.ndarray] = None):

    tree: EvalTreeNode = self._T_rq.get_eval_tree()
    T_rq = tree.value.matrix()

    error = self._D @ (self._reference - T_rq @ self._query)

    if lhs is None:
      return error

    lhs = -lhs @ self._D @ se3op.point2fs(T_rq @ self._query)
    jacs = self._T_rq.compute_jacs(lhs, tree)

    return error, jacs