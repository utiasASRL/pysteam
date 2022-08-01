import numpy as np
import numpy.linalg as npla

from ...evaluable import Evaluable, Node, Jacobians
from ...evaluable import se3 as se3ev, vspace as vspaceev
from ..const_vel.evaluators import jinv_velocity
from ..const_acc.evaluators import compose_curlyhat
from .helper import getQ, getTran
from .variable import Time, Variable


class PoseInterpolator(Evaluable):

  def __init__(self, time: Time, knot1: Variable, knot2: Variable, ad: np.ndarray) -> None:
    super().__init__()

    self._knot1: Variable = knot1
    self._knot2: Variable = knot2

    # calculate time constants
    T = (knot2.time - knot1.time).seconds
    tau = (time - knot1.time).seconds
    kappa = (knot2.time - time).seconds

    Q_tau = getQ(tau, ad)
    Q_T = getQ(T, ad)
    Tran_kappa = getTran(kappa, ad)
    Tran_tau = getTran(tau, ad)
    Tran_T = getTran(T, ad)

    Omega = Q_tau @ Tran_kappa.T @ npla.inv(Q_T)
    Lambda = Tran_tau - Omega @ Tran_T

    ## construct computation graph
    T1 = self._knot1.pose
    w1 = self._knot1.velocity
    dw1 = self._knot1.acceleration
    T2 = self._knot2.pose
    w2 = self._knot2.velocity
    dw2 = self._knot2.acceleration
    # get relative matrix info
    T_21 = se3ev.compose_rinv(T2, T1)
    # get se3 algebra of relative matrix
    xi_21 = se3ev.tran2vec(T_21)
    #
    gamma11 = w1
    gamma12 = dw1
    gamma20 = xi_21
    gamma21 = jinv_velocity(xi_21, w2)
    gamma22 = vspaceev.add(vspaceev.smult(compose_curlyhat(jinv_velocity(xi_21, w2), w2), -0.5),
                           jinv_velocity(xi_21, dw2))
    # calculate interpolated relative se3 algebra
    _t1 = vspaceev.mmult(gamma11, Lambda[:6, 6:12])
    _t2 = vspaceev.mmult(gamma12, Lambda[:6, 12:])
    _t3 = vspaceev.mmult(gamma20, Omega[:6, :6])
    _t4 = vspaceev.mmult(gamma21, Omega[:6, 6:12])
    _t5 = vspaceev.mmult(gamma22, Omega[:6, 12:])
    xi_i1 = vspaceev.add(_t1, vspaceev.add(_t2, vspaceev.add(_t3, vspaceev.add(_t4, _t5))))
    # calculate interpolated relative transformation matrix
    T_i1 = se3ev.vec2tran(xi_i1)
    # compose to get global transform
    T_i0 = se3ev.compose(T_i1, T1)
    # resulting evaluable
    self._T_i0 = T_i0

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
    return self._T_i0.forward()

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    self._T_i0.backward(lhs, node, jacs)
