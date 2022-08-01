from __future__ import annotations

import numpy as np
import numpy.linalg as npla
from typing import Dict, List

from pylgmath import Transformation

from ...evaluable import Evaluable
from ...evaluable import se3 as se3ev, vspace as vspaceev
from ...problem import StateVector, L2LossFunc, StaticNoiseModel, CostTerm, WeightedLeastSquareCostTerm
from ...solver import Covariance
from ..interface import Interface as TrajInterface
from .pose_interpolator import PoseInterpolator
from .prior_factor import PriorFactor
from .variable import Time, Variable
from .velocity_interpolator import VelocityInterpolator
from .helper import getQinv


class Interface(TrajInterface):
  """The trajectory class wraps a set of state variables to provide an interface that allows for continuous-time pose
  interpolation.
  """

  def __init__(self, qcd: np.ndarray = np.ones((6, 1))) -> None:
    assert qcd.shape == (6,), "qcd must be a (6,) vector"
    self._qcd: np.ndarray = qcd

    self._knots: Dict[int, Variable] = dict()
    self._ordered_nsecs_valid = True
    self._ordered_nsecs: np.ndarray = np.array([])

    self._pose_prior_factor = None
    self._velocity_prior_factor = None
    self._acceleration_prior_factor = None

  def add_knot(self, time: Time, T_k0: Evaluable, w_0k_ink: Evaluable, dw_0k_ink: Evaluable) -> None:
    assert not time.nanosecs in self._knots, "Knot already exists."
    self._knots[time.nanosecs] = Variable(time, T_k0, w_0k_ink, dw_0k_ink)
    self._ordered_nsecs_valid = False

  def add_pose_prior(self, time: Time, T_k0: Transformation, cov: np.ndarray) -> None:
    assert self._knots, "Knot dictionary is empty."
    assert time.nanosecs in self._knots.keys(), "No knot at provided time."
    assert self._pose_prior_factor is None, "A pose prior already exists."

    # get knot at specified time
    knot = self._knots[time.nanosecs]
    assert knot.pose.active, "Adding prior to locked pose."

    # set up loss function, noise model, and error function
    loss_func = L2LossFunc()
    noise_model = StaticNoiseModel(cov, "covariance")
    error_func = se3ev.se3_error(knot.pose, T_k0)
    # create cost term
    self._pose_prior_factor = WeightedLeastSquareCostTerm(error_func, noise_model, loss_func)

  def add_velocity_prior(self, time: Time, w_0k_ink: np.ndarray, cov: np.ndarray) -> None:
    """Add a unary velocity prior factor at a knot time."""
    assert self._knots, "Knot dictionary is empty."
    assert time.nanosecs in self._knots.keys(), "No knot at provided time."
    assert self._velocity_prior_factor is None, "A velocity prior already exists."

    # get knot at specified time
    knot = self._knots[time.nanosecs]
    assert knot.velocity.active, "Adding prior to locked velocity."

    # set up loss function, noise model, and error function
    loss_func = L2LossFunc()
    noise_model = StaticNoiseModel(cov, "covariance")
    error_func = vspaceev.vspace_error(knot.velocity, w_0k_ink)
    # create cost term
    self._velocity_prior_factor = WeightedLeastSquareCostTerm(error_func, noise_model, loss_func)

  def add_acceleration_prior(self, time: Time, dw_0k_ink: np.ndarray, cov: np.ndarray) -> None:
    """Add a unary acceleration prior factor at a knot time."""
    assert self._knots, "Knot dictionary is empty."
    assert time.nanosecs in self._knots.keys(), "No knot at provided time."
    assert self._acceleration_prior_factor is None, "A acceleration prior already exists."

    # get knot at specified time
    knot = self._knots[time.nanosecs]
    assert knot.acceleration.active, "Adding prior to locked acceleration."

    # set up loss function, noise model, and error function
    loss_func = L2LossFunc()
    noise_model = StaticNoiseModel(cov, "covariance")
    error_func = vspaceev.vspace_error(knot.acceleration, dw_0k_ink)
    # create cost term
    self._acceleration_prior_factor = WeightedLeastSquareCostTerm(error_func, noise_model, loss_func)

  def get_prior_cost_terms(self) -> List[CostTerm]:
    """Get binary cost terms associated with the prior for active parts of the trajectory."""
    cost_terms = []

    if not self._knots:
      return cost_terms

    if self._pose_prior_factor is not None:
      cost_terms.append(self._pose_prior_factor)
    if self._velocity_prior_factor is not None:
      cost_terms.append(self._velocity_prior_factor)
    if self._acceleration_prior_factor is not None:
      cost_terms.append(self._acceleration_prior_factor)

    loss_func = L2LossFunc()

    if not self._ordered_nsecs_valid:
      self._ordered_nsecs = np.array(sorted(self._knots.keys()))
      self._ordered_nsecs_valid = True

    for t in range(1, len(self._ordered_nsecs)):
      # get knots
      knot1 = self._knots[self._ordered_nsecs[t - 1]]
      knot2 = self._knots[self._ordered_nsecs[t]]

      if (knot1.pose.active or knot1.velocity.active or knot1.acceleration.active or knot2.pose.active or
          knot2.velocity.active or knot2.acceleration.active):

        # generate 12 x 12 information matrix for GP prior factor
        Qinv = getQinv((knot2.time - knot1.time).seconds, self._qcd)
        noise_model = StaticNoiseModel(Qinv, 'information')

        # create cost term
        error_func = PriorFactor(knot1, knot2)
        cost_term = WeightedLeastSquareCostTerm(error_func, noise_model, loss_func)

        cost_terms.append(cost_term)

    return cost_terms

  def get_pose_interpolator(self, time: Time):
    """Get pose evaluator at specified time stamp."""
    assert self._knots, "Knot dictionary is empty."

    if not self._ordered_nsecs_valid:
      self._ordered_nsecs = np.array(sorted(self._knots.keys()))
      self._ordered_nsecs_valid = True

    idx = np.searchsorted(self._ordered_nsecs, time.nanosecs)

    # request time exactly on a knot
    if idx < len(self._ordered_nsecs) and self._ordered_nsecs[idx] == time.nanosecs:
      return self._knots[self._ordered_nsecs[idx]].pose

    if idx == 0 or idx == len(self._ordered_nsecs):
      # request time before the first knot
      if idx == 0:
        raise NotImplementedError("Requested time before first knot.")
        # start_knot = self._knots[self._ordered_nsecs[0]]
        # T_t_k_eval = PoseExtrapolator(start_knot.velocity, time - start_knot.time)
        # return se3ev.compose(T_t_k_eval, start_knot.pose)
      # request time after the last knot
      else:
        raise NotImplementedError("Requested time after last knot.")
        # end_knot = self._knots[self._ordered_nsecs[-1]]
        # T_t_k_eval = PoseExtrapolator(end_knot.velocity, time - end_knot.time)
        # return se3ev.compose(T_t_k_eval, end_knot.pose)

    # request time between two knots, needs interpolation
    knot1 = self._knots[self._ordered_nsecs[idx - 1]]
    knot2 = self._knots[self._ordered_nsecs[idx]]
    return PoseInterpolator(time, knot1, knot2)

  def get_velocity_interpolator(self, time: Time):
    """Get velocity evaluator at specified time stamp."""
    assert self._knots, "Knot dictionary is empty."

    if not self._ordered_nsecs_valid:
      self._ordered_nsecs = np.array(sorted(self._knots.keys()))
      self._ordered_nsecs_valid = True

    idx = np.searchsorted(self._ordered_nsecs, time.nanosecs)

    # request time exactly on a knot
    if idx < len(self._ordered_nsecs) and self._ordered_nsecs[idx] == time.nanosecs:
      return self._knots[self._ordered_nsecs[idx]].velocity

    if idx == 0 or idx == len(self._ordered_nsecs):
      # request time before first knot
      if idx == 0:
        raise NotImplementedError("Requested time before first knot.")
        # return self._knots[self._ordered_nsecs[0]].velocity
      # request time after last knot
      else:
        raise NotImplementedError("Requested time after last knot.")
        # return self._knots[self._ordered_nsecs[-1]].velocity

    # request time needs interpolation
    knot1 = self._knots[self._ordered_nsecs[idx - 1]]
    knot2 = self._knots[self._ordered_nsecs[idx]]
    return VelocityInterpolator(time, knot1, knot2)