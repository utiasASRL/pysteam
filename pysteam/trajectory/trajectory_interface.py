from __future__ import annotations

import numpy as np
from typing import Dict, List

from pylgmath import Transformation, se3op
from ..evaluatable import Evaluatable
from ..evaluatable.se3 import SE3StateVar, LogMapEvaluator, InverseEvaluator, ComposeEvaluator
from ..evaluatable.vspace import VSpaceStateVar, AdditionEvaluator, NegationEvaluator
from ..problem import L2LossFunc, StaticNoiseModel, CostTerm, WeightedLeastSquareCostTerm
from .trajectory_var import Time, TrajectoryVar
from .trajectory_prior_factor import TrajectoryPriorFactor
from .trajectory_pose_interpolator import PoseInterpolator
from .trajectory_velocity_interpolator import VelocityInterpolator
from .trajectory_const_vel_transform_eval import ConstVelTransformEvaluator


class TrajectoryInterface:
  """The trajectory class wraps a set of state variables to provide an interface that allows for continuous-time pose
  interpolation.
  """

  def __init__(self, Qc_inv: np.ndarray = np.eye(6), allow_extrapolation: bool = True) -> None:
    self._Qc_inv: np.ndarray = Qc_inv
    self._allow_extrapolation: bool = allow_extrapolation

    # prior factors
    self._knots: Dict[int, TrajectoryVar] = dict()
    self._ordered_nsecs_valid = True
    self._ordered_nsecs: np.ndarray = np.array([])

    self._pose_prior_factor = None
    self._velocity_prior_factor = None

  def add_knot(self,
               *,
               knot: TrajectoryVar = None,
               time: Time = None,
               T_k0: Evaluatable = None,
               w_0k_ink: Evaluatable = None) -> None:
    if knot is not None:
      assert not knot.time.nanosecs in self._knots, "Knot already exists."
      self._knots[knot.time.nanosecs] = knot
      self._ordered_nsecs_valid = False
    elif time is not None and T_k0 is not None and w_0k_ink is not None:
      assert not time.nanosecs in self._knots, "Knot already exists."
      self._knots[time.nanosecs] = TrajectoryVar(time, T_k0, w_0k_ink)
      self._ordered_nsecs_valid = False
    else:
      raise ValueError("Invalid input combination.")

  def add_pose_prior(self, time: Time, T_21: Transformation, cov: np.ndarray) -> None:
    """Add a unary pose prior factor at a knot time.
    Note that only a single pose prior should exist on a trajectory, adding a second will overwrite the first.
    """
    assert self._knots, "Knot dictionary is empty."
    assert time.nanosecs in self._knots.keys(), "No knot at provided time."

    # get knot at specified time
    knot = self._knots[time.nanosecs]
    assert knot.pose.active, "Adding prior to locked pose."

    # set up loss function, noise model, and error function
    loss_func = L2LossFunc()
    noise_model = StaticNoiseModel(cov, "covariance")
    T_21_var = SE3StateVar(T_21, locked=True)
    error_func = LogMapEvaluator(ComposeEvaluator(T_21_var, InverseEvaluator(knot.pose)))

    # create cost term
    self._pose_prior_factor = WeightedLeastSquareCostTerm(error_func, noise_model, loss_func)

  def add_velocity_prior(self, time: Time, w_0k_ink: np.ndarray, cov: np.ndarray) -> None:
    """Add a unary velocity prior factor at a knot time.
    Note that only a single velocity prior should exist on a trajectory, adding a second will overwrite the first.
    """
    assert self._knots, "Knot dictionary is empty."
    assert time.nanosecs in self._knots.keys(), "No knot at provided time."

    # get knot at specified time
    knot = self._knots[time.nanosecs]
    assert knot.velocity.active, "Adding prior to locked velocity."

    # set up loss function, noise model, and error function
    loss_func = L2LossFunc()
    noise_model = StaticNoiseModel(cov, "covariance")
    w_0k_ink_var = VSpaceStateVar(w_0k_ink, locked=True)
    error_func = AdditionEvaluator(w_0k_ink_var, NegationEvaluator(knot.velocity))

    # create cost term
    self._velocity_prior_factor = WeightedLeastSquareCostTerm(error_func, noise_model, loss_func)

  def get_prior_cost_terms(self) -> List[CostTerm]:
    """Get binary cost terms associated with the prior for active parts of the trajectory."""
    cost_terms = []

    if not self._knots:
      return cost_terms

    if self._pose_prior_factor is not None:
      cost_terms.append(self._pose_prior_factor)

    if self._velocity_prior_factor is not None:
      cost_terms.append(self._velocity_prior_factor)

    loss_func = L2LossFunc()

    if not self._ordered_nsecs_valid:
      self._ordered_nsecs = np.array(sorted(self._knots.keys()))
      self._ordered_nsecs_valid = True

    for t in range(1, len(self._ordered_nsecs)):
      # get knots
      knot1 = self._knots[self._ordered_nsecs[t - 1]]
      knot2 = self._knots[self._ordered_nsecs[t]]

      if (knot1.pose.active or knot1.velocity.active or knot2.pose.active or knot2.velocity.active):

        # generate 12 x 12 information matrix for GP prior factor
        Qi_inv = np.zeros((12, 12))
        one_over_dt = 1 / (knot2.time - knot1.time).seconds
        one_over_dt2 = one_over_dt * one_over_dt
        one_over_dt3 = one_over_dt2 * one_over_dt
        Qi_inv[:6, :6] = 12 * one_over_dt3 * self._Qc_inv
        Qi_inv[:6, 6:] = Qi_inv[6:, :6] = -6 * one_over_dt2 * self._Qc_inv
        Qi_inv[6:, 6:] = 4 * one_over_dt * self._Qc_inv
        noise_model = StaticNoiseModel(Qi_inv, 'information')

        # create cost term
        error_func = TrajectoryPriorFactor(knot1, knot2)
        cost_term = WeightedLeastSquareCostTerm(error_func, noise_model, loss_func)

        cost_terms.append(cost_term)

    return cost_terms

  def get_pose_interpolator(self, time: Time):
    """Get transform evaluator at specified time stamp."""
    assert self._knots, "Knot dictionary is empty."

    if not self._ordered_nsecs_valid:
      self._ordered_nsecs = np.array(sorted(self._knots.keys()))
      self._ordered_nsecs_valid = True

    idx = np.searchsorted(self._ordered_nsecs, time.nanosecs)

    # request time exactly on a knot
    if idx < len(self._ordered_nsecs) and self._ordered_nsecs[idx] == time.nanosecs:
      return self._knots[self._ordered_nsecs[idx]].pose

    if idx == 0 or idx == len(self._ordered_nsecs):
      if not self._allow_extrapolation:
        raise ValueError("Query time out-of-range with extrapolation disallowed.")
      # request time before the first knot
      elif idx == 0:
        start_knot = self._knots[self._ordered_nsecs[0]]
        T_t_k_eval = ConstVelTransformEvaluator(start_knot.velocity, time - start_knot.time)
        return ComposeEvaluator(T_t_k_eval, start_knot.pose)
      # request time after the last knot
      else:
        end_knot = self._knots[self._ordered_nsecs[-1]]
        T_t_k_eval = ConstVelTransformEvaluator(end_knot.velocity, time - end_knot.time)
        return ComposeEvaluator(T_t_k_eval, end_knot.pose)

    # request time between two knots, needs interpolation
    knot1 = self._knots[self._ordered_nsecs[idx - 1]]
    knot2 = self._knots[self._ordered_nsecs[idx]]
    return PoseInterpolator(time, knot1, knot2)

  def get_velocity_interpolator(self, time: Time):
    """Get velocity at specified time stamp. TODO: make this an evaluator."""
    assert self._knots, "Knot dictionary is empty."

    if not self._ordered_nsecs_valid:
      self._ordered_nsecs = np.array(sorted(self._knots.keys()))
      self._ordered_nsecs_valid = True

    idx = np.searchsorted(self._ordered_nsecs, time.nanosecs)

    # request time exactly on a knot
    if idx < len(self._ordered_nsecs) and self._ordered_nsecs[idx] == time.nanosecs:
      return self._knots[self._ordered_nsecs[idx]].velocity

    if idx == 0 or idx == len(self._ordered_nsecs):
      if not self._allow_extrapolation:
        raise ValueError("Query time out-of-range with extrapolation disallowed.")
      # request time before first knot
      elif idx == 0:
        return self._knots[self._ordered_nsecs[0]].velocity
      # request time after last knot
      else:
        return self._knots[self._ordered_nsecs[-1]].velocity

    # request time needs interpolation
    knot1 = self._knots[self._ordered_nsecs[idx - 1]]
    knot2 = self._knots[self._ordered_nsecs[idx]]
    return VelocityInterpolator(time, knot1, knot2)
