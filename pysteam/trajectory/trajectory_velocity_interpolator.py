from ..evaluable import Evaluable, Node
from ..evaluable import se3 as se3ev, vspace as vspaceev
from .trajectory_var import Time, TrajectoryVar
from .evaluators import j_velocity, jinv_velocity


class VelocityInterpolator(Evaluable):

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
    # calculate interpolated relative se3 algebra
    _t1 = vspaceev.smult(w1, self._lambda12)
    _t2 = vspaceev.smult(xi_21, self._psi11)
    _t3 = vspaceev.smult(jinv_velocity(xi_21, w2), self._psi12)
    xi_i1 = vspaceev.add(_t1, vspaceev.add(_t2, _t3))
    # calculate interpolated relative se3 algebra
    _s1 = vspaceev.smult(w1, self._lambda22)
    _s2 = vspaceev.smult(xi_21, self._psi21)
    _s3 = vspaceev.smult(jinv_velocity(xi_21, w2), self._psi22)
    xi_it_linear = vspaceev.add(_s1, vspaceev.add(_s2, _s3))
    xi_it = j_velocity(xi_i1, xi_it_linear)
    self._xi_it = xi_it

  @property
  def active(self) -> bool:
    return ((self._knot1.pose.active or self._knot1.velocity.active) or
            (self._knot2.pose.active or self._knot2.velocity.active))

  def forward(self) -> Node:
    return self._xi_it.forward()

  def backward(self, lhs, node):
    return self._xi_it.backward(lhs, node)
