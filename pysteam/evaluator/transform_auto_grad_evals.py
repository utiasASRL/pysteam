from typing import Dict
import numpy as np

from pylgmath import se3op, Transformation
from . import AutoGradEvaluator, EvalTreeNode
from ..state import TransformStateVar, LandmarkStateVar


class TransformEvaluator(AutoGradEvaluator):
  pass


class LogMapEvaluator(TransformEvaluator):
  """Evaluator for the logarithmic map of a transformation matrix."""

  def __init__(self, transform: TransformEvaluator) -> None:
    super().__init__()
    self._transform: TransformEvaluator = transform

  def is_active(self):
    return self._transform.is_active()

  def get_eval_tree(self):
    child = self._transform.get_eval_tree()
    value = child.value.vec()
    root = EvalTreeNode(value, child)
    return root

  def compute_jacs(self, lhs, tree):
    jacs = dict()
    if self._transform.is_active():
      lhs = lhs @ se3op.vec2jacinv(tree.value)
      jacs = self._transform.compute_jacs(lhs, tree.children[0])

    return jacs


class InverseTransformEvaluator(TransformEvaluator):
  """Evaluator for the inverse of a transformation matrix"""

  def __init__(self, transform: TransformEvaluator) -> None:
    super().__init__()
    self._transform: TransformEvaluator = transform

  def is_active(self):
    return self._transform.is_active()

  def get_eval_tree(self) -> EvalTreeNode:
    child = self._transform.get_eval_tree()
    value = child.value.inverse()
    root = EvalTreeNode(value, child)
    return root

  def compute_jacs(self, lhs, tree) -> Dict:
    jacs = dict()
    if self._transform.is_active():
      lhs = -lhs @ tree.value.adjoint()
      jacs = self._transform.compute_jacs(lhs, tree.children[0])

    return jacs


class FixedTransformEvaluator(TransformEvaluator):
  """Simple transform evaluator for a fixed transformation."""

  def __init__(self, transform: Transformation) -> None:
    super().__init__()
    self._transform: Transformation = transform

  def is_active(self):
    return False

  def get_eval_tree(self):
    return EvalTreeNode(self._transform)

  def compute_jacs(self, lhs, tree):
    return dict()


class TransformStateEvaluator(TransformEvaluator):
  """Simple transform evaluator for a transformation state variable."""

  def __init__(self, transform: TransformStateVar):
    super().__init__()
    self._transform: TransformStateVar = transform

  def is_active(self):
    return not self._transform.is_locked()

  def get_eval_tree(self):
    return EvalTreeNode(self._transform.get_value())

  def compute_jacs(self, lhs, tree):
    jacs = dict()

    if not self._transform.is_locked():
      jacs = {self._transform.get_key(): lhs}

    return jacs


class ComposeTransformEvaluator(TransformEvaluator):
  """Evaluator for the composition of transformation matrices."""

  def __init__(self, transform1: TransformEvaluator, transform2: TransformEvaluator):
    super().__init__()
    self._transform1: TransformEvaluator = transform1
    self._transform2: TransformEvaluator = transform2

  def is_active(self):
    return self._transform1.is_active() or self._transform2.is_active()

  def get_eval_tree(self):
    child1 = self._transform1.get_eval_tree()
    child2 = self._transform2.get_eval_tree()

    value = child1.value @ child2.value
    root = EvalTreeNode(value, child1, child2)
    return root

  def compute_jacs(self, lhs, tree):
    jacs = dict()

    if self._transform1.is_active():
      jacs = self._transform1.compute_jacs(lhs, tree.children[0])

    if self._transform2.is_active():
      lhs = lhs @ tree.children[0].value.adjoint()
      jacs2 = self._transform2.compute_jacs(lhs, tree.children[1])
      jacs = self.merge_jacs(jacs, jacs2)

    return jacs


class ComposeInverseTransformEvaluator(TransformEvaluator):
  """Evaluator for the composition of two transformation matrices (with one inverted)."""

  def __init__(self, transform1, transform2):
    super().__init__()
    self._transform1 = transform1
    self._transform2 = transform2

  def is_active(self):
    return self._transform1.is_active() or self._transform2.is_active()

  def get_eval_tree(self):
    child1 = self._transform1.get_eval_tree()
    child2 = self._transform2.get_eval_tree()

    value = child1.value @ child2.value.inverse()
    root = EvalTreeNode(value, child1, child2)
    return root

  def compute_jacs(self, lhs, tree):
    jacs = dict()

    if self._transform1.is_active():
      jacs = self._transform1.compute_jacs(lhs, tree.children[0])

    if self._transform2.is_active():
      tf_ba = tree.children[0].value @ tree.children[1].value.inverse()
      lhs = -lhs @ tf_ba.adjoint()
      jacs2 = self._transform2.compute_jacs(lhs, tree.children[1])
      jacs = self.merge_jacs(jacs, jacs2)

    return jacs


class ComposeLandmarkEvaluator(TransformEvaluator):
  """Evaluator for the composition of a transformation evaluator and landmark state."""

  def __init__(self, transform: TransformEvaluator, landmark: LandmarkStateVar):
    super().__init__()
    self._transform: TransformEvaluator = transform
    self._landmark: LandmarkStateVar = landmark

  def is_active(self):
    return self._transform.is_active() or not self._landmark.is_locked()

  def get_eval_tree(self):

    transform_child = self._transform.get_eval_tree()
    landmark_leaf = EvalTreeNode(self._landmark.get_value())

    value = transform_child.value.matrix() @ landmark_leaf.value
    root = EvalTreeNode(value, transform_child, landmark_leaf)
    return root

  def compute_jacs(self, lhs, tree):
    jacs = dict()

    if self._transform.is_active():
      homogeneous = tree.value
      new_lhs = lhs @ se3op.point2fs(homogeneous)
      jacs = self._transform.compute_jacs(new_lhs, tree.children[0])

    if not self._landmark.is_locked():
      land_jac = np.zeros((4, 6))
      land_jac = tree.children[0].value.matrix()[:4, :3]
      jacs = self.merge_jacs(jacs, {self._landmark.get_key(), lhs @ land_jac})

    return jacs
