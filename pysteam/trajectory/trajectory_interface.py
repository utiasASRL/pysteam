from __future__ import annotations

import numpy as np
from typing import Dict, List

from pylgmath import Transformation, se3op
from ..state import VectorSpaceStateVar
from ..evaluator import TransformEvaluator, TransformErrorEval, VectorSpaceErrorEval
from ..problem import L2LossFunc, StaticNoiseModel, CostTerm, WeightedLeastSquareCostTerm
from . import Time, TrajectoryVar, TrajectoryPriorFactor, TrajectoryInterpPoseEval


class TrajectoryInterface:
  """The trajectory class wraps a set of state variables to provide an interface that allows for continuous-time pose
  interpolation.
  """

  def __init__(self, Qc_inv: np.ndarray = np.eye(6), allow_extrapolation: bool = False) -> None:
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
               T_k0: TransformEvaluator = None,
               velocity: VectorSpaceStateVar = None) -> None:
    if knot is not None:
      assert not knot.time.nanosecs in self._knots, "Knot already exists."
      self._knots[knot.time.nanosecs] = knot
      self._ordered_nsecs_valid = False
    elif time is not None and T_k0 is not None and velocity is not None:
      assert not time.nanosecs in self._knots, "Knot already exists."
      self._knots[time.nanosecs] = TrajectoryVar(time, T_k0, velocity)
      self._ordered_nsecs_valid = False
    else:
      raise ValueError("Invalid input combination.")

  def add_pose_prior(self, time: Time, pose: Transformation, cov: np.ndarray) -> None:
    """Add a unary pose prior factor at a knot time.
    Note that only a single pose prior should exist on a trajectory, adding a second will overwrite the first.
    """
    assert self._knots, "Knot dictionary is empty."
    assert time.nanosecs in self._knots.keys(), "No knot at provided time."

    # get knot at specified time
    knot = self._knots[time.nanosecs]
    assert knot.pose.is_active(), "Adding prior to locked pose."

    # set up loss function, noise model, and error function
    loss_func = L2LossFunc()
    noise_model = StaticNoiseModel(cov, "covariance")
    error_func = TransformErrorEval(meas_T_21=pose, T_21=knot.pose)

    # create cost term
    self._pose_prior_factor = WeightedLeastSquareCostTerm(error_func, noise_model, loss_func)

  def add_velocity_prior(self, time: Time, velocity: np.ndarray, cov: np.ndarray) -> None:
    """Add a unary velocity prior factor at a knot time.
    Note that only a single velocity prior should exist on a trajectory, adding a second will overwrite the first.
    """
    assert self._knots, "Knot dictionary is empty."
    assert time.nanosecs in self._knots.keys(), "No knot at provided time."

    # get knot at specified time
    knot = self._knots[time.nanosecs]
    assert not knot.velocity.is_locked(), "Adding prior to locked velocity."

    # set up loss function, noise model, and error function
    loss_func = L2LossFunc()
    noise_model = StaticNoiseModel(cov, "covariance")
    error_func = VectorSpaceErrorEval(velocity, knot.velocity)

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

      if (knot1.pose.is_active() or not knot1.velocity.is_locked() or knot2.pose.is_active() or
          not knot2.velocity.is_locked()):

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

  def get_interp_pose_eval(self, time: Time) -> TrajectoryInterpPoseEval:
    """Get transform evaluator at specified time stamp."""
    assert self._knots, "Knot dictionary is empty."

    if not self._ordered_nsecs_valid:
      self._ordered_nsecs = np.array(sorted(self._knots.keys()))
      self._ordered_nsecs_valid = True

    idx = np.searchsorted(self._ordered_nsecs, time.nanosecs)
    if self._ordered_nsecs[idx] == time.nanosecs:
      return self._knots[self._ordered_nsecs[idx]].pose

    if idx == 0 or idx == len(self._ordered_nsecs):
      raise NotImplementedError("Extrapolation not implemented yet")

    return TrajectoryInterpPoseEval(time, self._knots[self._ordered_nsecs[idx - 1]],
                                    self._knots[self._ordered_nsecs[idx]])

  def get_interp_velocity(self, time: Time) -> np.ndarray:
    """Get velocity at specified time stamp. TODO: make this an evaluator."""
    assert self._knots, "Knot dictionary is empty."

    if not self._ordered_nsecs_valid:
      self._ordered_nsecs = np.array(sorted(self._knots.keys()))
      self._ordered_nsecs_valid = True

    idx = np.searchsorted(self._ordered_nsecs, time.nanosecs)
    if self._ordered_nsecs[idx] == time.nanosecs:
      return self._knots[self._ordered_nsecs[idx]].velocity.get_value()

    if idx == 0 or idx == len(self._ordered_nsecs):
      raise NotImplementedError("Extrapolation not implemented yet")

    # interpolate
    knot1 = self._knots[self._ordered_nsecs[idx - 1]]
    knot2 = self._knots[self._ordered_nsecs[idx]]

    # calculate time constants
    tau = (time - knot1.time).seconds
    T = (knot2.time - knot1.time).seconds
    ratio = tau / T
    ratio2 = ratio * ratio
    ratio3 = ratio2 * ratio

    # calculate 'psi' interpolation values
    psi11 = 3.0 * ratio2 - 2.0 * ratio3
    psi12 = tau * (ratio2 - ratio)
    psi21 = 6.0 * (ratio - ratio2) / T
    psi22 = 3.0 * ratio2 - 2.0 * ratio

    # calculate (some of the) 'lambda' interpolation values
    lambda12 = tau - T * psi11 - psi12
    lambda22 = 1.0 - T * psi21 - psi22

    # get relative matrix info
    T_21 = knot2.pose.evaluate() @ knot1.pose.evaluate().inverse()

    # get se3 algebra of relative matrix (and cache it)
    xi_21 = T_21.vec()

    # calculate the 6x6 associated Jacobian (and cache it)
    J_21_inv = se3op.vec2jacinv(xi_21)

    # calculate interpolated relative se3 algebra
    xi_i1 = lambda12 * knot1.velocity.get_value() + psi11 * xi_21 + psi12 * J_21_inv @ knot2.velocity.get_value()

    # calculate the 6x6 associated Jacobian
    J_t1 = se3op.vec2jac(xi_i1)

    # calculate interpolated relative se3 algebra
    xi_it = J_t1 @ (lambda22 * knot1.velocity.get_value() + psi21 * xi_21 +
                    psi22 * J_21_inv @ knot2.velocity.get_value())

    return xi_it