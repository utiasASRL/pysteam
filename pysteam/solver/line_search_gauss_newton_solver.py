import numpy.linalg as npla

from ..problem import Problem
from .gauss_newton_solver import GaussNewtonSolver


class LineSearchGaussNewtonSolver(GaussNewtonSolver):

  def __init__(self, problem: Problem, **parameters) -> None:
    super().__init__(problem, **parameters)
    # override parameters
    self._parameters.update({"backtrack_multiplier": 0.5, "max_backtrack_steps": 10})
    self._parameters.update(**parameters)

  def linearize_solve_and_update(self):

    # initialize new cost with old cost in case of failure
    new_cost = self._prev_cost

    # build the system
    A, b = self._problem.build_gauss_newton_terms()
    grad_norm = npla.norm(b)  # compute gradient norm for termination check

    # solve the system
    perturbation = self.solve_gauss_newton(A, b)

    # apply update with line search
    backtrack_coeff = 1.0
    num_backtrack = 0
    step_success = False
    while num_backtrack < self._parameters["max_backtrack_steps"]:
      proposed_cost = self.propose_update(backtrack_coeff * perturbation)
      if proposed_cost <= self._prev_cost:
        self.accept_proposed_state()
        new_cost = proposed_cost
        step_success = True
        break
      else:
        self.reject_proposed_state()
        backtrack_coeff *= self._parameters["backtrack_multiplier"]

      num_backtrack += 1

    # print report line if verbose option is enabled
    if (self._parameters["verbose"]):
      print(f"Iteration: {self._curr_iteration:4}  -  Cost: {new_cost:10.4f}  -  Search Coeff: {backtrack_coeff:6.3f}")

    return step_success, new_cost, grad_norm