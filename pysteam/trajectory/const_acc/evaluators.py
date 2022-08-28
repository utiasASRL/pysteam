import numpy as np

from pylgmath import se3op

from ...evaluable import Evaluable, Node, Jacobians


class ComposeCurlyhatEvaluator(Evaluable):
  """Evaluator for the composition curlyhat(x) @ y."""

  def __init__(self, x: Evaluable, y: Evaluable):
    super().__init__()
    self._x: Evaluable = x
    self._y: Evaluable = y

  @property
  def active(self) -> bool:
    return self._x.active or self._y.active

  @property
  def related_var_keys(self) -> set:
    return self._x.related_var_keys | self._y.related_var_keys

  def forward(self) -> Node:
    x = self._x.forward()
    y = self._y.forward()
    value = se3op.curlyhat(x.value) @ y.value
    return Node(value, x, y)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    x, y = node.children

    if self._x.active:
      self._x.backward(-lhs @ se3op.curlyhat(y.value), x, jacs)

    if self._y.active:
      self._y.backward(lhs @ se3op.curlyhat(x.value), y, jacs)


compose_curlyhat = ComposeCurlyhatEvaluator