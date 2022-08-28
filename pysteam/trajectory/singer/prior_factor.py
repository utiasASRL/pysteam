import numpy as np

from ...evaluable import Evaluable, Node, Jacobians
from ...evaluable import se3 as se3ev, vspace as vspaceev
from ..const_vel.evaluators import jinv_velocity
from ..const_acc.evaluators import compose_curlyhat
from .variable import Variable


class PriorFactor(Evaluable):

  def __init__(self, knot1: Variable, knot2: Variable, ad: np.ndarray) -> None:
    super().__init__()

    self._knot1: Variable = knot1
    self._knot2: Variable = knot2

    # constants
    dt = (self._knot2.time - self._knot1.time).seconds
    ad_inv = 1.0 / ad  # diagonal matrix
    alpha = np.diag(ad)
    alpha_inv = np.diag(ad_inv)

    # construct computation graph
    T1 = self._knot1.pose
    w1 = self._knot1.velocity
    dw1 = self._knot1.acceleration
    T2 = self._knot2.pose
    w2 = self._knot2.velocity
    dw2 = self._knot2.acceleration

    eye = np.eye(6)
    expon = np.diag(np.exp(-dt * ad))
    C1 = alpha_inv @ alpha_inv @ (dt * alpha - eye + expon)
    C2 = alpha_inv @ (eye - expon)
    C3 = expon

    # get relative matrix info
    T_21 = se3ev.compose_rinv(T2, T1)
    # get se3 algebra of relative matrix
    xi_21 = se3ev.tran2vec(T_21)

    # pose error
    _t1 = xi_21
    _t2 = vspaceev.smult(w1, -dt)
    _t3 = vspaceev.mmult(dw1, -C1)
    self._ep = vspaceev.add(_t1, vspaceev.add(_t2, _t3))

    # velocity error
    _w1 = jinv_velocity(xi_21, w2)
    _w2 = vspaceev.neg(w1)
    _w3 = vspaceev.mmult(dw1, -C2)
    self._ev = vspaceev.add(_w1, vspaceev.add(_w2, _w3))

    # acceleration error
    _dw1 = vspaceev.smult(compose_curlyhat(jinv_velocity(xi_21, dw2), dw2), -0.5)
    _dw2 = jinv_velocity(xi_21, dw2)
    _dw3 = vspaceev.mmult(dw1, -C3)
    self._ea = vspaceev.add(_dw1, vspaceev.add(_dw2, _dw3))

  @property
  def active(self) -> bool:
    return ((self._knot1.pose.active or self._knot1.velocity.active or self._knot1.acceleration.active) or
            (self._knot2.pose.active or self._knot2.velocity.active or self._knot2.acceleration.active))

  @property
  def related_var_keys(self) -> set:
    keys = set()
    keys |= self._knot1.pose.related_var_keys
    keys |= self._knot1.velocity.related_var_keys
    keys |= self._knot1.acceleration.related_var_keys
    keys |= self._knot2.pose.related_var_keys
    keys |= self._knot2.velocity.related_var_keys
    keys |= self._knot2.acceleration.related_var_keys
    return keys

  def forward(self) -> Node:
    ep = self._ep.forward()
    ev = self._ev.forward()
    ea = self._ea.forward()
    e = np.concatenate((ep.value, ev.value, ea.value), axis=0)
    return Node(e, ep, ev, ea)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    ep, ev, ea = node.children
    self._ep.backward(lhs[..., :6], ep, jacs)
    self._ev.backward(lhs[..., 6:12], ev, jacs)
    self._ea.backward(lhs[..., 12:], ea, jacs)