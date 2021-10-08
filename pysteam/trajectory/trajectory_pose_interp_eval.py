import numpy as np

from pylgmath import se3op, Transformation

from ..evaluator import EvalTreeNode, AutoGradEvaluator, TransformEvaluator
from . import Time, TrajectoryVar


class TrajectoryInterpPoseEval(TransformEvaluator):

  def __init__(self, time: Time, knot1: TrajectoryVar, knot2: TrajectoryVar) -> None:
    super().__init__()

    self._knot1: TrajectoryVar = knot1
    self._knot2: TrajectoryVar = knot2

    # calculate time constants
    tau = (time - knot1.time).seconds
    T = (knot2.time - knot1.time).seconds
    ratio = tau / T
    ratio2 = ratio * ratio
    ratio3 = ratio2 * ratio

    # calculate 'psi' interpolation values
    self._psi11 = 3.0 * ratio2 - 2.0 * ratio3
    self._psi12 = tau * (ratio2 - ratio)
    self._psi21 = 6.0 * (ratio - ratio2) / T
    self._psi22 = 3.0 * ratio2 - 2.0 * ratio

    # calculate 'lambda' interpolation values
    self._lambda11 = 1.0 - self._psi11
    self._lambda12 = tau - T * self._psi11 - self._psi12
    self._lambda21 = -self._psi21
    self._lambda22 = 1.0 - T * self._psi21 - self._psi22

  def is_active(self) -> bool:
    return ((self._knot1.pose.is_active() or not self._knot1.velocity.is_locked()) or
            (self._knot2.pose.is_active() or not self._knot2.velocity.is_locked()))

  def get_eval_tree(self):
    # evaluate sub-trees
    transform1 = self._knot1.pose.get_eval_tree()
    transform2 = self._knot2.pose.get_eval_tree()

    # get relative matrix info
    T_21: Transformation = transform2.value @ transform1.value.inverse()

    # get se3 algebra of relative matrix
    xi_21: np.ndarray = T_21.vec()

    # calculate the 6x6 associated Jacobian
    J_21_inv = se3op.vec2jacinv(xi_21)

    # calculate interpolated relative se3 algebra
    xi_i1 = (self._lambda12 * self._knot1.velocity.get_value() + self._psi11 * xi_21 +
             self._psi12 * J_21_inv @ self._knot2.velocity.get_value())

    # calculate interpolated relative transformation matrix
    T_i1 = Transformation(xi_ab=xi_i1)

    # interpolated relative transform - new root node
    root = EvalTreeNode(T_i1 @ transform1.value, transform1, transform2)

    return root

  def compute_jacs(self, lhs, tree):
    jacs = dict()

    if not self.is_active():
      return jacs

    # evaluate sub-trees
    transform1 = tree.children[0]
    transform2 = tree.children[1]

    # get relative matrix info
    T_21: Transformation = transform2.value @ transform1.value.inverse()

    # get se3 algebra of relative matrix
    xi_21: np.ndarray = T_21.vec()

    # calculate the 6x6 associated Jacobian
    J_21_inv = se3op.vec2jacinv(xi_21)

    # calculate interpolated relative se3 algebra
    xi_i1 = (self._lambda12 * self._knot1.velocity.get_value() + self._psi11 * xi_21 +
             self._psi12 * J_21_inv @ self._knot2.velocity.get_value())

    # calculate interpolated relative transformation matrix
    T_i1 = Transformation(xi_ab=xi_i1)

    # calculate the 6x6 Jacobian associated with the interpolated relative transformation matrix
    J_i1 = se3op.vec2jac(xi_i1)

    if self._knot1.pose.is_active() or self._knot2.pose.is_active():
      w = (self._psi11 * J_i1 @ J_21_inv +
           0.5 * self._psi12 * J_i1 @ se3op.curlyhat(self._knot2.velocity.get_value()) @ J_21_inv)

      if self._knot1.pose.is_active():
        jacobian = (-1) * w @ T_21.adjoint() + T_i1.adjoint()
        jacs1 = self._knot1.pose.compute_jacs(lhs @ jacobian, transform1)
        jacs = AutoGradEvaluator.merge_jacs(jacs, jacs1)

      if self._knot2.pose.is_active():
        jacs2 = self._knot2.pose.compute_jacs(lhs @ w, transform2)
        jacs = AutoGradEvaluator.merge_jacs(jacs, jacs2)

    # 6 x 6 Velocity Jacobian 1
    if not self._knot1.velocity.is_locked():
      jacs3 = {self._knot1.velocity.get_key(): self._lambda12 * lhs @ J_i1}
      jacs = AutoGradEvaluator.merge_jacs(jacs, jacs3)

    # 6 x 6 Velocity Jacobian 2
    if not self._knot2.velocity.is_locked():
      jacobian = self._psi12 * J_i1 @ J_21_inv
      jacs4 = {self._knot2.velocity.get_key(): lhs @ jacobian}
      jacs = AutoGradEvaluator.merge_jacs(jacs, jacs4)

    return jacs