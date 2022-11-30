from __future__ import annotations

from typing import Dict, List

import numpy as np
import numpy.linalg as npla

from ...evaluable import Evaluable
from ...evaluable import se3 as se3ev
from ...evaluable import vspace as vspaceev
from ...problem import (
    CostTerm,
    L2LossFunc,
    StateVector,
    StaticNoiseModel,
    WeightedLeastSquareCostTerm,
)
from ...solver import Covariance
from ..interface import Interface as TrajInterface
from ..time import Time
from . import VelocityInterpolator
from .variable import Variable


class Interface(TrajInterface):
    """The trajectory class wraps a set of state variables to provide an interface that allows for continuous-time pose
    interpolation.
    """

    def __init__(self, knot_spacing: Time) -> None:
        self._knot_spacing = knot_spacing
        self._knots: Dict[Time, Variable] = dict()  # six by one vector evaluable

    def get_prior_cost_terms(self) -> List[CostTerm]:
        """Get binary cost terms associated with the prior for active parts of the trajectory."""
        return []

    def get_state_vars(self) -> List[Evaluable]:
        """Get the state variables associated with the trajectory."""
        return [v.c for v in self._knots.values()]

    def get_velocity_interpolator(self, time: Time):
        """Get velocity evaluator at specified time stamp."""

        # get the relevant knot times in nanoseconds
        t2 = Time(
            nsecs=(
                self._knot_spacing.nanosecs
                * int(np.floor(time.nanosecs / self._knot_spacing.nanosecs))
            )
        )
        t1 = t2 - self._knot_spacing
        t3 = t2 + self._knot_spacing
        t4 = t3 + self._knot_spacing
        # print(f"t1: {t1.seconds}, t2: {t2.seconds}, t3: {t3.seconds}, t4: {t4.seconds}")

        knots = []
        for t in [t1, t2, t3, t4]:
            if not t in self._knots.keys():
                self._knots[t] = Variable(t, vspaceev.VSpaceStateVar(np.zeros((6, 1))))
            knots.append(self._knots[t])
        return VelocityInterpolator(time, *knots)
