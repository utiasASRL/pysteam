import numpy as np

from ...evaluable import Evaluable, Jacobians, Node
from ...evaluable import se3 as se3ev
from ...evaluable import vspace as vspaceev
from .evaluators import jinv_velocity
from .helper import getQ, getQinv, getTran
from .variable import Time, Variable


class PoseInterpolator(Evaluable):
    def __init__(self, time: Time, knot1: Variable, knot2: Variable) -> None:
        super().__init__()

        self._knot1: Variable = knot1
        self._knot2: Variable = knot2

        # calculate time constants
        T = (knot2.time - knot1.time).seconds
        tau = (time - knot1.time).seconds
        kappa = (knot2.time - time).seconds

        Q_tau = getQ(tau)
        Qinv_T = getQinv(T)
        Tran_kappa = getTran(kappa)
        Tran_tau = getTran(tau)
        Tran_T = getTran(T)

        Omega = Q_tau @ Tran_kappa.T @ Qinv_T
        Lambda = Tran_tau - Omega @ Tran_T

        ## construct computation graph
        T1 = self._knot1.pose
        w1 = self._knot1.velocity
        T2 = self._knot2.pose
        w2 = self._knot2.velocity
        # get relative matrix info
        T_21 = se3ev.compose_rinv(T2, T1)
        # get se3 algebra of relative matrix
        xi_21 = se3ev.tran2vec(T_21)
        #
        gamma11 = w1
        gamma20 = xi_21
        gamma21 = jinv_velocity(xi_21, w2)
        # calculate interpolated relative se3 algebra
        _t1 = vspaceev.smult(gamma11, Lambda[0, 1])
        _t2 = vspaceev.smult(gamma20, Omega[0, 0])
        _t3 = vspaceev.smult(gamma21, Omega[0, 1])
        xi_i1 = vspaceev.add(_t1, vspaceev.add(_t2, _t3))
        # calculate interpolated relative transformation matrix
        T_i1 = se3ev.vec2tran(xi_i1)
        # compose to get global transform
        T_i0 = se3ev.compose(T_i1, T1)
        # resulting evaluable
        self._T_i0 = T_i0

    @property
    def active(self) -> bool:
        return (self._knot1.pose.active or self._knot1.velocity.active) or (
            self._knot2.pose.active or self._knot2.velocity.active
        )

    @property
    def related_var_keys(self) -> set:
        keys = set()
        keys |= self._knot1.pose.related_var_keys
        keys |= self._knot1.velocity.related_var_keys
        keys |= self._knot2.pose.related_var_keys
        keys |= self._knot2.velocity.related_var_keys
        return keys

    def forward(self) -> Node:
        return self._T_i0.forward()

    def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
        self._T_i0.backward(lhs, node, jacs)
