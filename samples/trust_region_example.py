"""
A sample usage of the STEAM Engine library for testing various trust-region solvers on a divergent (for Gauss-Newton)
error metric.
"""
import numpy as np
from typing import Optional

from pysteam.evaluator import Evaluator
from pysteam.problem import StaticNoiseModel, L2LossFunc, WeightedLeastSquareCostTerm, OptimizationProblem
from pysteam.state import VectorSpaceStateVar
from pysteam.solver import GaussNewtonSolver, LineSearchGaussNewtonSolver, LevMarqGaussNewtonSolver, DoglegGaussNewtonSolver


class DivergenceErrorEval(Evaluator):
  """A simple error metric designed to test trust region methods.
    Implements a vector error in R^2, e[0] = x + 1, e[1] = -2*x^2 + x - 1.
    Minimum error is at zero. Notably vanilla Gauss Newton is unable to converge to the answer as a step near zero
    causes it to diverge.
  """

  def __init__(self, state_vec: VectorSpaceStateVar) -> None:
    super().__init__()
    assert state_vec.get_perturb_dim() == 1
    self._state_vec = state_vec

  def is_active(self) -> bool:
    return not self._state_vec.is_locked()

  def evaluate(self, lhs: Optional[np.ndarray] = None):
    x = self._state_vec.get_value()[0, 0]
    eval = np.empty((2, 1))
    eval[0, 0] = x + 1.0
    eval[1, 0] = -2.0 * x * x + x - 1.0

    if lhs is None:
      return eval

    jacs = dict()

    if not self._state_vec.is_locked():
      jac = np.empty((2, 1))
      jac[0, 0] = 1.0
      jac[1, 0] = -4.0 * x + 1.0
      jacs = {self._state_vec.get_key(): lhs @ jac}

    return eval, jacs


def setup_divergence_problem():
  # Create vector state variable
  initial = np.array([[10.0]])
  state_var = VectorSpaceStateVar(initial)

  # Setup noise model and loss function
  noise_model = StaticNoiseModel(np.eye(2))
  loss_func = L2LossFunc()

  # Setup cost term
  error_func = DivergenceErrorEval(state_var)
  cost_term = WeightedLeastSquareCostTerm(error_func, noise_model, loss_func)

  # Initialize problem
  problem = OptimizationProblem()
  problem.add_state_var(state_var)
  problem.add_cost_term(cost_term)

  return problem


def main():
  # Solve using Gauss-Newton Solver
  problem = setup_divergence_problem()
  solver = GaussNewtonSolver(problem, max_iterations=100)
  solver.optimize()
  print("Gauss Newton terminates from:", solver.termination_cause, "after", solver.curr_iteration, "iterations.")

  # Solve using Line Search Gauss-Newton Solver
  problem = setup_divergence_problem()
  solver = LineSearchGaussNewtonSolver(problem, max_iterations=100)
  solver.optimize()
  print("Line Search GN terminates from:", solver.termination_cause, "after", solver.curr_iteration, "iterations.")

  # Solve using Levenberg-Marquardt Solver
  problem = setup_divergence_problem()
  solver = LevMarqGaussNewtonSolver(problem, max_iterations=100)
  solver.optimize()
  print("Levenberg–Marquardt terminates from:", solver.termination_cause, "after", solver.curr_iteration, "iterations.")

  # Solve using Powell's Dogleg Solver
  problem = setup_divergence_problem()
  solver = DoglegGaussNewtonSolver(problem, max_iterations=100)
  solver.optimize()
  print("Powell's Dogleg terminates from:", solver.termination_cause, "after", solver.curr_iteration, "iterations.")


if __name__ == "__main__":
  main()