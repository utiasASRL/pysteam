import numpy as np

from pylgmath import se3op

from ..evaluable import Evaluable, Node, Jacobians


class RadialVelErrorEvaluator(Evaluable):
  """Evaluates radial velocity error."""

  def __init__(
      self,
      w_iv_inv: Evaluable,
      pv: np.ndarray,
      r: float,
  ) -> None:
    """
    Args:
      w_iv_inv: body-velocity of the moving frame (vehicle or sensor frame)
      pv: cartesian coordinate of the point in the moving frame
      r: measured radial velocity
    """
    super().__init__()

    self._w_iv_inv: Evaluable = w_iv_inv
    self._pv: np.ndarray = pv
    self._r: float = np.array([[r]])  # to 1x1 matrix for easier multiplication

    self._D = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])

  @property
  def active(self) -> bool:
    return self._w_iv_inv.active

  @property
  def related_var_keys(self) -> set:
    return self._w_iv_inv.related_var_keys

  def forward(self) -> Node:
    child = self._w_iv_inv.forward()

    numerator = self._pv.T @ self._D @ se3op.point2fs(self._pv, 1.0) @ child.value
    denominator = np.sqrt(self._pv.T @ self._pv)
    r = numerator / denominator
    value = self._r - r

    return Node(value, child)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._w_iv_inv.active:
      jac_unnorm = self._pv.T @ self._D @ se3op.point2fs(self._pv, 1.0)
      pv_norm = np.sqrt(self._pv.T @ self._pv)
      jac = jac_unnorm / pv_norm
      self._w_iv_inv.backward(-lhs @ jac, node.children[0], jacs)