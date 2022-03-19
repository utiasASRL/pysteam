import numpy as np

from pylgmath import se3op, Transformation

from ..evaluatable import Evaluatable, Node
from .trajectory_var import Time, TrajectoryVar


class PoseInterpolator(Evaluatable):

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

  @property
  def active(self) -> bool:
    return ((self._knot1.pose.active or self._knot1.velocity.active) or
            (self._knot2.pose.active or self._knot2.velocity.active))

  def forward(self) -> Node:
    # evaluate sub-trees
    pose_node1 = self._knot1.pose.forward()
    velocity_node1 = self._knot1.velocity.forward()
    pose_node2 = self._knot2.pose.forward()
    velocity_node2 = self._knot2.velocity.forward()

    # get relative matrix info
    T_21: Transformation = pose_node2.value @ pose_node1.value.inverse()

    # get se3 algebra of relative matrix
    xi_21: np.ndarray = T_21.vec()

    # calculate the 6x6 associated Jacobian
    J_21_inv = se3op.vec2jacinv(xi_21)

    # calculate interpolated relative se3 algebra
    xi_i1 = (self._lambda12 * velocity_node1.value + self._psi11 * xi_21 +
             self._psi12 * J_21_inv @ velocity_node2.value)

    # calculate interpolated relative transformation matrix
    T_i1 = Transformation(xi_ab=xi_i1)

    # interpolated relative transform - new root node
    return Node(T_i1 @ pose_node1.value, pose_node1, velocity_node1, pose_node2, velocity_node2)

  def backward(self, lhs, node):
    jacs = dict()

    pose_node1, velocity_node1, pose_node2, velocity_node2 = node.children

    # get relative matrix info
    T_21: Transformation = pose_node2.value @ pose_node1.value.inverse()

    # get se3 algebra of relative matrix
    xi_21: np.ndarray = T_21.vec()

    # calculate the 6x6 associated Jacobian
    J_21_inv = se3op.vec2jacinv(xi_21)

    # calculate interpolated relative se3 algebra
    xi_i1 = (self._lambda12 * velocity_node1.value + self._psi11 * xi_21 +
             self._psi12 * J_21_inv @ velocity_node2.value)

    # calculate interpolated relative transformation matrix
    T_i1 = Transformation(xi_ab=xi_i1)

    # calculate the 6x6 Jacobian associated with the interpolated relative transformation matrix
    J_i1 = se3op.vec2jac(xi_i1)

    if self._knot1.pose.active or self._knot2.pose.active:
      w = (self._psi11 * J_i1 @ J_21_inv + 0.5 * self._psi12 * J_i1 @ se3op.curlyhat(velocity_node2.value) @ J_21_inv)

      if self._knot1.pose.active:
        jacobian = (-1) * w @ T_21.adjoint() + T_i1.adjoint()
        jacs1 = self._knot1.pose.backward(lhs @ jacobian, pose_node1)
        jacs = self.merge_jacs(jacs, jacs1)

      if self._knot2.pose.active:
        jacs2 = self._knot2.pose.backward(lhs @ w, pose_node2)
        jacs = self.merge_jacs(jacs, jacs2)

    # 6 x 6 Velocity Jacobian 1
    if self._knot1.velocity.active:
      jacs3 = self._knot1.velocity.backward(self._lambda12 * lhs @ J_i1, velocity_node1)
      jacs = self.merge_jacs(jacs, jacs3)

    # 6 x 6 Velocity Jacobian 2
    if self._knot2.velocity.active:
      jacobian = self._psi12 * J_i1 @ J_21_inv
      jacs4 = self._knot2.velocity.backward(lhs @ jacobian, velocity_node2)
      jacs = self.merge_jacs(jacs, jacs4)

    return jacs