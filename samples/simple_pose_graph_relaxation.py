"""Example that loads and solves a relative pose graph problem.
"""
import numpy as np
from typing import List

from pylgmath import Transformation
from pysteam.problem import StaticNoiseModel, L2LossFunc, WeightedLeastSquareCostTerm, OptimizationProblem
from pysteam.solver import GaussNewtonSolver
from pysteam.state import TransformStateVar
from pysteam.evaluator import TransformErrorEval


class RelMeas:
  """Structure to store simulated relative transform measurements."""

  def __init__(self, idxA: int, idxB: int, meas_T_BA: Transformation) -> None:
    self.idxA = idxA
    self.idxB = idxB
    self.meas_T_BA = meas_T_BA


def main():
  ## Setup 'dataset'
  ## simulate a simple odometry-style dataset of relative poses (no loop closures).
  num_poses = 1000
  meas_collection: List[RelMeas] = []
  for i in range(1, num_poses):
    meas_vec = np.array([[-1.0, 0., 0., 0., 0., 0.01]]).T
    meas_collection.append(RelMeas(i - 1, i, Transformation(xi_ab=meas_vec)))

  ## Setup states
  poses = [TransformStateVar(Transformation()) for _ in range(num_poses)]
  poses[0].set_lock(True)  # lock first pose (otherwise entire solution is 'floating')

  ## Setup shared noise and loss function
  noise_model = StaticNoiseModel(np.eye(6))
  loss_func = L2LossFunc()

  ## Turn measurements into cost terms
  cost_terms = []
  for i in range(len(meas_collection)):
    state_A = poses[meas_collection[i].idxA]
    state_B = poses[meas_collection[i].idxB]
    meas_T_BA = meas_collection[i].meas_T_BA
    error_func = TransformErrorEval(meas_T_21=meas_T_BA, T_20=state_B, T_10=state_A)
    cost_terms.append(WeightedLeastSquareCostTerm(error_func, noise_model, loss_func))

  ## Make Optimization Problem
  problem = OptimizationProblem()
  problem.add_state_var(*poses)
  problem.add_cost_term(*cost_terms)

  ## Make solver and solve
  solver = GaussNewtonSolver(problem, verbose=True)
  solver.optimize()


if __name__ == "__main__":
  main()
