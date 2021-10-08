import numpy as np

from pylgmath import se3op, Transformation
from ..evaluator import Evaluator, AutoGradEvaluator
from . import TrajectoryVar


class TrajectoryPriorFactor(Evaluator):

  def __init__(self, knot1: TrajectoryVar, knot2: TrajectoryVar) -> None:
    super().__init__()

    self._knot1: TrajectoryVar = knot1
    self._knot2: TrajectoryVar = knot2

  def is_active(self):
    return ((self._knot1.pose.is_active() or not self._knot1.velocity.is_locked()) or
            (self._knot2.pose.is_active() or not self._knot2.velocity.is_locked()))

  def evaluate(self, lhs=None):
    tree1 = self._knot1.pose.get_eval_tree()
    tree2 = self._knot2.pose.get_eval_tree()

    T_21: Transformation = tree2.value @ tree1.value.inverse()
    xi_21 = T_21.vec()
    J_21_inv = se3op.vec2jacinv(xi_21)
    dt = (self._knot2.time - self._knot1.time).seconds

    error = np.empty((12, 1))
    error[:6] = xi_21 - dt * self._knot1.velocity.get_value()
    error[6:] = J_21_inv @ self._knot2.velocity.get_value() - self._knot1.velocity.get_value()
    if lhs is None:
      return error

    jacs = dict()

    if self._knot1.pose.is_active():
      Jinv_12: np.ndarray = J_21_inv @ T_21.adjoint()
      # construct Jacobian
      jacobian = np.empty((12, 6))
      jacobian[:6] = -Jinv_12
      jacobian[6:] = -0.5 * se3op.curlyhat(self._knot2.velocity.get_value()) @ Jinv_12

      # get Jacobians
      jacs1 = self._knot1.pose.compute_jacs(lhs @ jacobian, tree1)
      jacs = AutoGradEvaluator.merge_jacs(jacs, jacs1)

    if self._knot2.pose.is_active():
      jacobian = np.empty((12, 6))
      jacobian[:6] = J_21_inv
      jacobian[6:] = (0.5 * se3op.curlyhat(self._knot2.velocity.get_value()) @ J_21_inv)

      jacs2 = self._knot2.pose.compute_jacs(lhs @ jacobian, tree2)
      jacs = AutoGradEvaluator.merge_jacs(jacs, jacs2)

    if not self._knot1.velocity.is_locked():
      # construct Jacobian object
      jacobian = np.empty((12, 6))
      jacobian[:6] = -dt * np.eye(6)
      jacobian[6:] = -np.eye(6)
      jacs3 = {self._knot1.velocity.get_key(): lhs @ jacobian}
      jacs = AutoGradEvaluator.merge_jacs(jacs, jacs3)

    if not self._knot2.velocity.is_locked():
      # construct Jacobian object
      jacobian = np.empty((12, 6))
      jacobian[:6] = np.zeros((6, 6))
      jacobian[6:] = J_21_inv

      jacs4 = {self._knot2.velocity.get_key(): lhs @ jacobian}
      jacs = AutoGradEvaluator.merge_jacs(jacs, jacs4)

    return error, jacs
