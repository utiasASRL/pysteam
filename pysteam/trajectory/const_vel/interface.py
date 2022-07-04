from __future__ import annotations

import numpy as np
from typing import Dict, List

from pylgmath import Transformation

from ...evaluable import Evaluable
from ...evaluable import se3 as se3ev, vspace as vspaceev
from ...problem import OptimizationProblem, L2LossFunc, StaticNoiseModel, CostTerm, WeightedLeastSquareCostTerm
from ...solver import Covariance
from ..interface import Interface as TrajInterface
from .evaluators import state_error
from .pose_extrapolator import PoseExtrapolator
from .pose_interpolator import PoseInterpolator
from .prior_factor import PriorFactor
from .variable import Time, Variable
from .velocity_interpolator import VelocityInterpolator


class Interface(TrajInterface):
  """The trajectory class wraps a set of state variables to provide an interface that allows for continuous-time pose
  interpolation.
  """

  def __init__(self, Qc_inv: np.ndarray = np.eye(6)) -> None:
    self._Qc_inv: np.ndarray = Qc_inv

    # prior factors
    self._knots: Dict[int, Variable] = dict()
    self._ordered_nsecs_valid = True
    self._ordered_nsecs: np.ndarray = np.array([])

    self._pose_prior_factor = None
    self._velocity_prior_factor = None

  def add_knot(self, time: Time, T_k0: Evaluable, w_0k_ink: Evaluable) -> None:
    assert not time.nanosecs in self._knots, "Knot already exists."
    self._knots[time.nanosecs] = Variable(time, T_k0, w_0k_ink)
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
        start_knot = self._knots[self._ordered_nsecs[0]]
        T_t_k_eval = PoseExtrapolator(start_knot.velocity, time - start_knot.time)
        return se3ev.compose(T_t_k_eval, start_knot.pose)
      # request time after the last knot
      else:
        end_knot = self._knots[self._ordered_nsecs[-1]]
        T_t_k_eval = PoseExtrapolator(end_knot.velocity, time - end_knot.time)
        return se3ev.compose(T_t_k_eval, end_knot.pose)

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
        return self._knots[self._ordered_nsecs[0]].velocity
      # request time after last knot
      else:
        return self._knots[self._ordered_nsecs[-1]].velocity

    # request time needs interpolation
    knot1 = self._knots[self._ordered_nsecs[idx - 1]]
    knot2 = self._knots[self._ordered_nsecs[idx]]
    return VelocityInterpolator(time, knot1, knot2)

  def get_covariance(self, cov: Covariance, time: Time):
    """Get velocity evaluator at specified time stamp."""
    assert self._knots, "Knot dictionary is empty."

    if not self._ordered_nsecs_valid:
      self._ordered_nsecs = np.array(sorted(self._knots.keys()))
      self._ordered_nsecs_valid = True

    idx = np.searchsorted(self._ordered_nsecs, time.nanosecs)

    # request time exactly on a knot
    if idx < len(self._ordered_nsecs) and self._ordered_nsecs[idx] == time.nanosecs:
      T_k0 = self._knots[self._ordered_nsecs[idx]].pose
      w_0k_ink = self._knots[self._ordered_nsecs[idx]].velocity
      assert isinstance(T_k0, se3ev.SE3StateVar) and isinstance(w_0k_ink, vspaceev.VSpaceStateVar)
      return cov.query([T_k0, w_0k_ink])

    # extrapolate
    elif idx == 0 or idx == len(self._ordered_nsecs):
      knot = self._knots[self._ordered_nsecs[idx]] if idx == 0 else self._knots[self._ordered_nsecs[-1]]

      T_q0_var = se3ev.SE3StateVar(self.get_pose_interpolator(time).evaluate())
      w_0q_inq_var = vspaceev.VSpaceStateVar(self.get_velocity_interpolator(time).evaluate())
      query_trajectory = Interface(self._Qc_inv)
      query_trajectory.add_knot(knot.time, knot.pose, knot.velocity)
      query_trajectory.add_knot(time, T_q0_var, w_0q_inq_var)

      loss_func = L2LossFunc()

      knot_noise = StaticNoiseModel(cov.query([knot.pose, knot.velocity]), "covariance")
      knot_error = state_error(se3ev.se3_error(knot.pose, knot.pose.value),
                               vspaceev.vspace_error(knot.velocity, knot.velocity.value))
      knot_cost = WeightedLeastSquareCostTerm(knot_error, knot_noise, loss_func)

      problem = OptimizationProblem()
      problem.add_state_var(knot.pose, knot.velocity, T_q0_var, w_0q_inq_var)
      problem.add_cost_term(*query_trajectory.get_prior_cost_terms(), knot_cost)

      query_cov = Covariance(problem)
      return query_cov.query([T_q0_var, w_0q_inq_var])

    # interpolate
    else:
      knot1 = self._knots[self._ordered_nsecs[idx - 1]]
      knot2 = self._knots[self._ordered_nsecs[idx]]

      T_q0_var = se3ev.SE3StateVar(self.get_pose_interpolator(time).evaluate())
      w_0q_inq_var = vspaceev.VSpaceStateVar(self.get_velocity_interpolator(time).evaluate())
      query_trajectory = Interface(self._Qc_inv)
      query_trajectory.add_knot(knot1.time, knot1.pose, knot1.velocity)
      query_trajectory.add_knot(time, T_q0_var, w_0q_inq_var)
      query_trajectory.add_knot(knot2.time, knot2.pose, knot2.velocity)

      loss_func = L2LossFunc()

      knot1_noise = StaticNoiseModel(cov.query([knot1.pose, knot1.velocity]), "covariance")
      knot1_error = state_error(se3ev.se3_error(knot1.pose, knot1.pose.value),
                                vspaceev.vspace_error(knot1.velocity, knot1.velocity.value))
      knot1_cost = WeightedLeastSquareCostTerm(knot1_error, knot1_noise, loss_func)

      knot2_noise = StaticNoiseModel(cov.query([knot2.pose, knot2.velocity]), "covariance")
      knot2_error = state_error(se3ev.se3_error(knot2.pose, knot2.pose.value),
                                vspaceev.vspace_error(knot2.velocity, knot2.velocity.value))
      knot2_cost = WeightedLeastSquareCostTerm(knot2_error, knot2_noise, loss_func)

      problem = OptimizationProblem()
      problem.add_state_var(knot1.pose, knot1.velocity, T_q0_var, w_0q_inq_var, knot2.pose, knot2.velocity)
      problem.add_cost_term(*query_trajectory.get_prior_cost_terms(), knot1_cost, knot2_cost)

      query_cov = Covariance(problem)
      return query_cov.query([T_q0_var, w_0q_inq_var])
