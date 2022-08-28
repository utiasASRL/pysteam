import numpy as np

from ...evaluable import Evaluable, Node, Jacobians
from ...evaluable import se3 as se3ev, vspace as vspaceev
from .variable import Variable
from .evaluators import jinv_velocity


class PriorFactor(Evaluable):

  def __init__(self, knot1: Variable, knot2: Variable) -> None:
    super().__init__()

    self._knot1: Variable = knot1
    self._knot2: Variable = knot2

    # constants
    dt = (self._knot2.time - self._knot1.time).seconds

    # construct computation graph
    T1 = self._knot1.pose
    w1 = self._knot1.velocity
    T2 = self._knot2.pose
    w2 = self._knot2.velocity

    # get relative matrix info
    T_21 = se3ev.compose_rinv(T2, T1)
    # get se3 algebra of relative matrix
    xi_21 = se3ev.tran2vec(T_21)

    # pose error
    _t1 = xi_21
    _t2 = vspaceev.smult(w1, -dt)
    self._ep = vspaceev.add(_t1, _t2)

    # velocity error
    _w1 = jinv_velocity(xi_21, w2)
    _w2 = vspaceev.neg(w1)
    self._ev = vspaceev.add(_w1, _w2)

  @property
  def active(self) -> bool:
    return ((self._knot1.pose.active or self._knot1.velocity.active) or
            (self._knot2.pose.active or self._knot2.velocity.active))

  @property
  def related_var_keys(self) -> set:
    keys = set()
    keys |= self._knot1.pose.related_var_keys
    keys |= self._knot1.velocity.related_var_keys
    keys |= self._knot2.pose.related_var_keys
    keys |= self._knot2.velocity.related_var_keys
    return keys

  def forward(self) -> Node:
    ep = self._ep.forward()
    ev = self._ev.forward()
    e = np.concatenate((ep.value, ev.value), axis=0)
    return Node(e, ep, ev)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    ep, ev = node.children
    self._ep.backward(lhs[..., :6], ep, jacs)
    self._ev.backward(lhs[..., 6:12], ev, jacs)