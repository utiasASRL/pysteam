import numpy as np

from pylgmath import se3op, Transformation

from ..evaluatable import Evaluatable, Node
from ..evaluatable import se3 as se3ev, vspace as vspaceev
from .trajectory_var import Time, TrajectoryVar
from .evaluators import jinv_velocity


class PoseInterpolator(Evaluatable):

  def __init__(self, time: Time, knot1: TrajectoryVar, knot2: TrajectoryVar) -> None:
    super().__init__()

    self._knot1: TrajectoryVar = knot1
    self._knot2: TrajectoryVar = knot2

    # calculate time constants
    tau = (time - knot1.time).seconds
    T = (knot2.time - knot1.time).seconds
    ratio = tau / T
    ratio2 = ratio * ratio
    ratio3 = ratio2 * ratio

    # calculate 'psi' interpolation values
    self._psi11 = 3.0 * ratio2 - 2.0 * ratio3
    self._psi12 = tau * (ratio2 - ratio)
    self._psi21 = 6.0 * (ratio - ratio2) / T
    self._psi22 = 3.0 * ratio2 - 2.0 * ratio

    # calculate 'lambda' interpolation values
    self._lambda11 = 1.0 - self._psi11
    self._lambda12 = tau - T * self._psi11 - self._psi12
    self._lambda21 = -self._psi21
    self._lambda22 = 1.0 - T * self._psi21 - self._psi22

    ## construct computation graph
    T1 = self._knot1.pose
    w1 = self._knot1.velocity
    T2 = self._knot2.pose
    w2 = self._knot2.velocity
    # get relative matrix info
    T_21 = se3ev.compose_rinv(T2, T1)
    # get se3 algebra of relative matrix
    xi_21 = se3ev.tran2vec(T_21)
    # calculate the 6x6 associated Jacobian
    J_21_inv = se3ev.vec2jacinv(xi_21)
    # calculate interpolated relative se3 algebra
    _t1 = vspaceev.smult(w1, self._lambda12)
    _t2 = vspaceev.smult(xi_21, self._psi11)
    _t3 = vspaceev.smult(jinv_velocity(J_21_inv, w2), self._psi12)
    xi_i1 = vspaceev.add(_t1, vspaceev.add(_t2, _t3))
    # calculate interpolated relative transformation matrix
    T_i1 = se3ev.vec2tran(xi_i1)
    # compose to get global transform
    T_i0 = se3ev.compose(T_i1, T1)
    # resulting evaluable
    self._T_i0 = T_i0

  @property
  def active(self) -> bool:
    return ((self._knot1.pose.active or self._knot1.velocity.active) or
            (self._knot2.pose.active or self._knot2.velocity.active))

  def forward(self) -> Node:
    return self._T_i0.forward()

  def backward(self, lhs, node):
    return self._T_i0.backward(lhs, node)
