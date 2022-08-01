import numpy as np
import numpy.linalg as npla

from ...evaluable import Evaluable, Node, Jacobians
from ...evaluable import se3 as se3ev, vspace as vspaceev
from .variable import Time, Variable
from .evaluators import j_velocity, jinv_velocity, compose_curlyhat
from .helper import getQ, getTran


class VelocityInterpolator(Evaluable):

  def __init__(self, time: Time, knot1: Variable, knot2: Variable) -> None:
    super().__init__()

    self._knot1: Variable = knot1
    self._knot2: Variable = knot2

    # calculate time constants
    T = (knot2.time - knot1.time).seconds
    tau = (time - knot1.time).seconds
    kappa = (knot2.time - time).seconds

    Q_tau = getQ(tau)
    Q_T = getQ(T)
    Tran_kappa = getTran(kappa)
    Tran_tau = getTran(tau)
    Tran_T = getTran(T)

    Omega = Q_tau @ Tran_kappa.T @ npla.inv(Q_T)
    Lambda = Tran_tau - Omega @ Tran_T

    _omega11 = Omega[:6, :6]
    _omega12 = Omega[:6, 6:12]
    _omega13 = Omega[:6, 12:]
    _omega21 = Omega[6:12, :6]
    _omega22 = Omega[6:12, 6:12]
    _omega23 = Omega[6:12, 12:]
    _lambda12 = Lambda[:6, 6:12]
    _lambda13 = Lambda[:6, 12:]
    _lambda22 = Lambda[6:12, 6:12]
    _lambda23 = Lambda[6:12, 12:]

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
    # calculate interpolated relative se3 algebra
    _t1 = vspaceev.mmult(w1, _lambda12)
    _t2 = vspaceev.mmult(dw1, _lambda13)
    _t3 = vspaceev.mmult(xi_21, _omega11)
    _t4 = vspaceev.mmult(jinv_velocity(xi_21, w2), _omega12)
    _t51 = vspaceev.mmult(vspaceev.smult(compose_curlyhat(jinv_velocity(xi_21, w2), w2), -0.5), _omega13)
    _t52 = vspaceev.mmult(jinv_velocity(xi_21, dw2), _omega13)
    xi_i1 = vspaceev.add(_t1, vspaceev.add(_t2, vspaceev.add(_t3, vspaceev.add(_t4, vspaceev.add(_t51, _t52)))))

    _s1 = vspaceev.mmult(w1, _lambda22)
    _s2 = vspaceev.mmult(dw1, _lambda23)
    _s3 = vspaceev.mmult(xi_21, _omega21)
    _s4 = vspaceev.mmult(jinv_velocity(xi_21, w2), _omega22)
    _s51 = vspaceev.mmult(vspaceev.smult(compose_curlyhat(jinv_velocity(xi_21, w2), w2), -0.5), _omega23)
    _s52 = vspaceev.mmult(jinv_velocity(xi_21, dw2), _omega23)
    xi_it_linear = vspaceev.add(_s1, vspaceev.add(_s2, vspaceev.add(_s3, vspaceev.add(_s4, vspaceev.add(_s51, _s52)))))
    xi_it = j_velocity(xi_i1, xi_it_linear)
    self._xi_it = xi_it

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
    return self._xi_it.forward()

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    self._xi_it.backward(lhs, node, jacs)
