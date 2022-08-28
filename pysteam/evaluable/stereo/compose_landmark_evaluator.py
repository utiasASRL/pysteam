import numpy as np

from pylgmath import se3op

from ..evaluable import Evaluable, Node, Jacobians


class ComposeLandmarkEvaluator(Evaluable):
  """Evaluator for the composition of a transformation evaluator and landmark state."""

  def __init__(self, transform: Evaluable, landmark: Evaluable):
    super().__init__()
    self._transform: Evaluable = transform
    self._landmark: Evaluable = landmark

  @property
  def active(self) -> bool:
    return self._transform.active or self._landmark.active

  @property
  def related_var_keys(self) -> set:
    return self._transform.related_var_keys | self._landmark.related_var_keys

  def forward(self) -> Node:
    transform_child = self._transform.forward()
    landmark_child = self._landmark.forward()

    value = transform_child.value.matrix() @ landmark_child.value

    return Node(value, transform_child, landmark_child)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    transform_child, landmark_child = node.children

    if self._transform.active:
      homogeneous = node.value
      new_lhs = lhs @ se3op.point2fs(homogeneous)
      self._transform.backward(new_lhs, transform_child, jacs)

    if self._landmark.active:
      land_jac = transform_child.value.matrix()[:4, :3]
      self._landmark.backward(lhs @ land_jac, landmark_child, jacs)
