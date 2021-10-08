import abc
import numpy as np
import timeit
from typing import Any, Dict
from copy import deepcopy

from ..problem import OptimizationProblem
from ..state import StateVector


class Solver(abc.ABC):

  def __init__(self, opt_prob: OptimizationProblem, **parameters: Dict[str, Any]) -> None:
    self._problem = opt_prob
    self._state_vector = StateVector()
    # solver parameters
    self._parameters = {
        "verbose": False,
        "max_iterations": 100,
        "absolute_cost_threshold": 0.0,
        "absolute_cost_change_threshold": 1e-4,
        "relative_cost_change_threshold": 1e-4,
    }
    self._parameters.update(**parameters)

    # set up state vector -- add all states that are not locked to vector
    for state_var in self._problem.get_state_vars():
      if not state_var.is_locked():
        self._state_vector.add_state_var(state_var)

    # create a backup state vector
    self._state_vector_backup = deepcopy(self._state_vector)

    # initial internal state
    self._term = "NOT YET TERMINATED"
    self._curr_iteration = 0
    self._solver_converged = False
    self._curr_cost = self._prev_cost = self._problem.cost()
    self._pending_proposed_state = False

  @property
  def parameters(self) -> Dict[str, Any]:
    return self._parameters

  @property
  def curr_iteration(self) -> int:
    return self._curr_iteration

  @property
  def coverged(self) -> bool:
    return self._solver_converged

  @property
  def termination_cause(self) -> str:
    return self._term

  def optimize(self) -> None:
    # setup timer
    start = timeit.default_timer()

    # optimization loop
    while not self._solver_converged:
      self.iterate()

    # log
    if self._parameters["verbose"]:
      end = timeit.default_timer()
      print("Total Optimization Time: {0:.4f} seconds".format(end - start))

  def iterate(self) -> None:
    # check colver converged
    if self._solver_converged:
      return

    # log on first iteration
    if self._parameters["verbose"] and self._curr_iteration == 0:
      print("Begin Optimization")
      print("------------------")
      print("Number of States: ", self._state_vector.get_num_states())
      print("Number of Cost Terms: ", self._problem.get_num_of_cost_terms())
      print("Initial Cost: ", self._curr_cost)

    # update iteration number
    self._curr_iteration += 1

    # record previous iteration cost
    self._prev_cost = self._curr_cost

    # perform algorithm specific update
    step_success, self._curr_cost, grad_norm = self.linearize_solve_and_update()

    # check termination criteria
    if not step_success and np.abs(grad_norm) < 1e-6:
      self._term = "CONVERGED ZERO GRADIENT"
      self._solver_converged = True
    elif not step_success:
      self._term = "STEP UNSUCCESSFUL"
      self._solver_converged = True
      raise RuntimeError("Unable to produce a 'successful' step.")
    elif self._curr_iteration >= self._parameters["max_iterations"]:
      self._term = "MAX ITERATIONS"
      self._solver_converged = True
    elif self._curr_cost <= self._parameters["absolute_cost_threshold"]:
      self._term = "CONVERGED ABSOLUTE CHANGE"
      self._solver_converged = True
    elif np.abs(self._prev_cost - self._curr_cost) <= self._parameters["absolute_cost_change_threshold"]:
      self._term = "CONVERGED ABSOLUTE CHANGE"
      self._solver_converged = True
    elif (np.abs(self._prev_cost - self._curr_cost) /
          self._prev_cost) <= self._parameters["relative_cost_change_threshold"]:
      self._term = "CONVERGED RELATIVE CHANGE"
      self._solver_converged = True

    # log on final iteration
    if self._parameters["verbose"] and self._solver_converged:
      print("Termination Cause: ", self._term)

  def propose_update(self, perturbation: np.ndarray) -> float:
    # check that an update is not already pending
    if self._pending_proposed_state:
      raise RuntimeError("There is already a pending update, accept or reject.")

    # make a copy of the state vector on first backup
    self._state_vector_backup.copy_values(self._state_vector)

    # update the states with perturbation
    self._state_vector.update(perturbation)
    self._pending_proposed_state = True

    # test new cost
    return self._problem.cost()

  def accept_proposed_state(self):
    # check that an update has been proposed
    if not self._pending_proposed_state:
      raise RuntimeError("Must call propose_update before accepting the update.")

    # switch flag, accepting the update
    self._pending_proposed_state = False

  def reject_proposed_state(self):
    # check that an update has been proposed
    if not self._pending_proposed_state:
      raise RuntimeError("Must call propose_update before rejecting the update.")

    # revert to previous state
    self._state_vector.copy_values(self._state_vector_backup)

    # switch flag, ready for new proposal
    self._pending_proposed_state = False

  @abc.abstractmethod
  def linearize_solve_and_update(self) -> None:
    """Build the system, solve for a step size and direction, and update the state"""
