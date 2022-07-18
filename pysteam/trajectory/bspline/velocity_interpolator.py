import numpy as np
from pyrsistent import v

from ...evaluable import Evaluable, Node, Jacobians
from ...evaluable import se3 as se3ev, vspace as vspaceev
from .variable import Time, Variable
from ..time import Time


class VelocityInterpolator(Evaluable):

  def __init__(self, time: Time, k1: Variable, k2: Variable, k3: Variable, k4: Variable) -> None:
    super().__init__()

    self._k1: Variable = k1
    self._k2: Variable = k2
    self._k3: Variable = k3
    self._k4: Variable = k4

    # k2 is the center time (doesn't matter secs or nanosec, since its relative)
    t = time.nanosecs
    b = self._k2.time.nanosecs
    bp1 = self._k3.time.nanosecs

    B = np.array([
        [1., 4., 1., 0.],
        [-3., 0., 3., 0.],
        [3., -6., 3., 0.],
        [-1., 3., -3., 1.],
    ]) / 6.

    ratio = (t - b) / (bp1 - b)
    u = np.array([[1., ratio, ratio**2, ratio**3]]).T

    self._w = (B.T @ u).flatten()

  @property
  def active(self) -> bool:
    return ((self._k1.c.active or self._k2.c.active) or (self._k3.c.active or self._k4.c.active))

  @property
  def related_var_keys(self) -> set:
    return self._k1.c.related_var_keys | self._k2.c.related_var_keys | self._k3.c.related_var_keys | self._k4.c.related_var_keys

  def forward(self) -> Node:
    k1 = self._k1.c.forward()
    k2 = self._k2.c.forward()
    k3 = self._k3.c.forward()
    k4 = self._k4.c.forward()
    value = self._w[0] * k1.value + self._w[1] * k2.value + self._w[2] * k3.value + self._w[3] * k4.value
    return Node(value, k1, k2, k3, k4)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._k1.c.active:
      self._k1.c.backward(lhs * self._w[0], node.children[0], jacs)
    if self._k2.c.active:
      self._k2.c.backward(lhs * self._w[1], node.children[1], jacs)
    if self._k3.c.active:
      self._k3.c.backward(lhs * self._w[2], node.children[2], jacs)
    if self._k4.c.active:
      self._k4.c.backward(lhs * self._w[3], node.children[3], jacs)
