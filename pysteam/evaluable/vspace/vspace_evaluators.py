import numpy as np

from ..evaluable import Evaluable, Node, Jacobians


class NegationEvaluator(Evaluable):
  """Evaluator for the negation of a vector."""

  def __init__(self, value: Evaluable) -> None:
    super().__init__()
    self._value: Evaluable = value

  @property
  def active(self) -> bool:
    return self._value.active

  @property
  def related_var_keys(self) -> set:
    return self._value.related_var_keys

  def forward(self) -> Node:
    child = self._value.forward()
    value = -child.value
    return Node(value, child)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._value.active:
      self._value.backward(-lhs, node.children[0], jacs)


neg = NegationEvaluator


class AdditionEvaluator(Evaluable):
  """Evaluator for the addition of two vectors."""

  def __init__(self, lhs: Evaluable, rhs: Evaluable) -> None:
    super().__init__()
    self._lhs: Evaluable = lhs
    self._rhs: Evaluable = rhs

  @property
  def active(self) -> bool:
    return self._lhs.active or self._rhs.active

  @property
  def related_var_keys(self) -> set:
    return self._lhs.related_var_keys | self._rhs.related_var_keys

  def forward(self) -> Node:
    lhs = self._lhs.forward()
    rhs = self._rhs.forward()
    value = lhs.value + rhs.value
    return Node(value, lhs, rhs)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._lhs.active:
      self._lhs.backward(lhs, node.children[0], jacs)
    if self._rhs.active:
      self._rhs.backward(lhs, node.children[1], jacs)


add = AdditionEvaluator


class ScalarMultEvaluator(Evaluable):

  def __init__(self, value: Evaluable, scalar: float) -> None:
    super().__init__()
    self._value: Evaluable = value
    self._scalar: float = scalar

  @property
  def active(self) -> bool:
    return self._value.active

  @property
  def related_var_keys(self) -> set:
    return self._value.related_var_keys

  def forward(self) -> Node:
    child = self._value.forward()
    value = self._scalar * child.value
    return Node(value, child)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._value.active:
      self._value.backward(self._scalar * lhs, node.children[0], jacs)


smult = ScalarMultEvaluator


class MatrixMultEvaluator(Evaluable):

  def __init__(self, value: Evaluable, matrix: np.ndarray) -> None:
    super().__init__()
    self._value: Evaluable = value
    self._matrix: np.ndarray = matrix

  @property
  def active(self) -> bool:
    return self._value.active

  @property
  def related_var_keys(self) -> set:
    return self._value.related_var_keys

  def forward(self) -> Node:
    child = self._value.forward()
    value = self._matrix @ child.value
    return Node(value, child)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._value.active:
      self._value.backward(lhs @ self._matrix, node.children[0], jacs)


mmult = MatrixMultEvaluator


class VSpaceErrorEvaluator(Evaluable):
  """Error evaluator for a vector space."""

  def __init__(self, value: Evaluable, value_meas: np.ndarray) -> None:
    super().__init__()
    self._value: Evaluable = value
    self._value_meas: np.ndarray = value_meas

  @property
  def active(self) -> bool:
    return self._value.active

  @property
  def related_var_keys(self) -> set:
    return self._value.related_var_keys

  def forward(self) -> Node:
    child = self._value.forward()
    value = self._value_meas - child.value
    return Node(value, child)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._value.active:
      self._value.backward(-lhs, node.children[0], jacs)


vspace_error = VSpaceErrorEvaluator