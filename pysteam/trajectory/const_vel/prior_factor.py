import numpy as np

from pylgmath import se3op, Transformation

from ...evaluable import Evaluable, Node, Jacobians
from .variable import Variable


class PriorFactor(Evaluable):

  def __init__(self, knot1: Variable, knot2: Variable) -> None:
    super().__init__()

    self._knot1: Variable = knot1
    self._knot2: Variable = knot2

  @property
  def active(self) -> bool:
    return ((self._knot1.pose.active or self._knot1.velocity.active) or
            (self._knot2.pose.active or self._knot2.velocity.active))

  def forward(self) -> Node:
    pose_node1 = self._knot1.pose.forward()
    velocity_node1 = self._knot1.velocity.forward()
    pose_node2 = self._knot2.pose.forward()
    velocity_node2 = self._knot2.velocity.forward()

    T_21: Transformation = pose_node2.value @ pose_node1.value.inverse()
    xi_21 = T_21.vec()
    J_21_inv = se3op.vec2jacinv(xi_21)
    dt = (self._knot2.time - self._knot1.time).seconds

    error = np.empty((12, 1))
    error[:6] = xi_21 - dt * velocity_node1.value
    error[6:] = J_21_inv @ velocity_node2.value - velocity_node1.value

    return Node(error, pose_node1, velocity_node1, pose_node2, velocity_node2)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    pose_node1, velocity_node1, pose_node2, velocity_node2 = node.children

    T_21: Transformation = pose_node2.value @ pose_node1.value.inverse()
    xi_21 = T_21.vec()
    J_21_inv = se3op.vec2jacinv(xi_21)
    dt = (self._knot2.time - self._knot1.time).seconds

    if self._knot1.pose.active:
      Jinv_12: np.ndarray = J_21_inv @ T_21.adjoint()
      jacobian = np.empty((12, 6))
      jacobian[:6] = -Jinv_12
      jacobian[6:] = -0.5 * se3op.curlyhat(velocity_node2.value) @ Jinv_12

      self._knot1.pose.backward(lhs @ jacobian, pose_node1, jacs)

    if self._knot2.pose.active:
      jacobian = np.empty((12, 6))
      jacobian[:6] = J_21_inv
      jacobian[6:] = (0.5 * se3op.curlyhat(velocity_node2.value) @ J_21_inv)

      self._knot2.pose.backward(lhs @ jacobian, pose_node2, jacs)

    if self._knot1.velocity.active:
      jacobian = np.empty((12, 6))
      jacobian[:6] = -dt * np.eye(6)
      jacobian[6:] = -np.eye(6)

      self._knot1.velocity.backward(lhs @ jacobian, velocity_node1, jacs)

    if self._knot2.velocity.active:
      jacobian = np.empty((12, 6))
      jacobian[:6] = np.zeros((6, 6))
      jacobian[6:] = J_21_inv

      self._knot2.velocity.backward(lhs @ jacobian, velocity_node2, jacs)
