import numpy as np
import numpy.linalg as npla

from .state_vector import StateVector
from .problem import Problem


class SlidingWindowFilter(Problem):

  def __init__(self):
    self._variables = {}  # list of state variables, note that dict preserves order
    self._related_var_keys = {}  # dict of related variables (check if we can stop re-linearizing)
    self._cost_terms = []  # list of cost terms

    # marginalized values
    self._fixed_A = None
    self._fixed_b = None

  def add_variable(self, *variables):
    for variable in variables:
      self._variables[variable.key] = [False, variable]
      self._related_var_keys[variable.key] = {variable.key}

  def marginalize_variable(self, *variables):
    if len(variables) == 0:
      return

    for variable in variables:
      assert variable.key in self._variables.keys(), "variable key not found in variables dictionary"
      self._variables[variable.key][0] = True

    # need to cache all fixed variables!
    fixed_state_vector = StateVector()
    state_vector = StateVector()

    to_remove = []
    fixed = True  # check that variables to be marginalized are at the first
    for _, var in self._variables.values():
      related_keys = self._related_var_keys[var.key]
      if all([self._variables[key][0] for key in related_keys]):
        assert fixed, "marginalized variables must be at the first"
        fixed_state_vector.add_state_var(var)
        to_remove.append(var.key)
      else:
        fixed = False
      state_vector.add_state_var(var)

    # build first to remove the fixed variables
    fixed_state_size = fixed_state_vector.get_state_size()
    state_size = state_vector.get_state_size()

    # build the full linear system
    A = np.zeros((state_size, state_size))
    b = np.zeros((state_size, 1))

    # remove the fixed cost terms
    active_cost_terms = []
    for cost_term in self._cost_terms:
      related_keys = cost_term.related_var_keys
      if all([self._variables[key][0] for key in related_keys]):
        cost_term.build_gauss_newton_terms(state_vector, A, b)
        # print("removing cost term:", cost_term.name)
      else:
        active_cost_terms.append(cost_term)
    self._cost_terms = active_cost_terms

    # add the cached terms (always top-left block)
    if self._fixed_A is not None:
      A[:self._fixed_A.shape[0], :self._fixed_A.shape[1]] += self._fixed_A
      b[:self._fixed_b.shape[0]] += self._fixed_b

    # marginalize the fixed variables
    if fixed_state_size > 0:
      A00 = A[:fixed_state_size, :fixed_state_size]
      A10 = A[fixed_state_size:, :fixed_state_size]
      A11 = A[fixed_state_size:, fixed_state_size:]
      b0 = b[:fixed_state_size]
      b1 = b[fixed_state_size:]
      self._fixed_A = A11 - A10 @ npla.inv(A00) @ A10.T
      self._fixed_b = b1 - A10 @ npla.inv(A00) @ b0
    else:
      self._fixed_A = A
      self._fixed_b = b

    # remove the fixed variables
    for key in to_remove:
      # print("removing variable:", self._variables[key][1].name)
      related_keys = self._related_var_keys[key].copy()
      for related_key in related_keys:
        self._related_var_keys[related_key].remove(key)
      del self._related_var_keys[key]
      del self._variables[key]

  def add_cost_term(self, *cost_terms):
    for cost_term in cost_terms:
      # add cost term to the list of cost terms
      self._cost_terms.append(cost_term)
      # add related variables to the dictionary
      related_var_keys = cost_term.related_var_keys
      for key in related_var_keys:
        assert key in self._related_var_keys.keys(), "related variable key not found in related variables dictionary"
        self._related_var_keys[key].update(related_var_keys)

  def cost(self):
    return sum([x.cost() for x in self._cost_terms])

  def get_num_of_cost_terms(self):
    return len(self._cost_terms)

  def get_state_vector(self):
    self.marginalize_state_vector = StateVector()
    self.active_state_vector = StateVector()
    self.state_vector = StateVector()

    marginalize = True  # check that variables to be marginalized are at the first
    for mg, var in self._variables.values():
      if mg:
        assert marginalize, "marginalized variables must be at the first"
        self.marginalize_state_vector.add_state_var(var)
      else:
        mg = False
        self.active_state_vector.add_state_var(var)
      self.state_vector.add_state_var(var)

    return self.active_state_vector

  def build_gauss_newton_terms(self):

    # build first to remove the fixed variables
    marginalize_state_size = self.marginalize_state_vector.get_state_size()
    state_size = self.state_vector.get_state_size()

    # build the full linear system
    A = np.zeros((state_size, state_size))
    b = np.zeros((state_size, 1))
    for cost_term in self._cost_terms:
      cost_term.build_gauss_newton_terms(self.state_vector, A, b)

    # add the cached terms (always top-left block)
    if self._fixed_A is not None:
      A[:self._fixed_A.shape[0], :self._fixed_A.shape[1]] += self._fixed_A
      b[:self._fixed_b.shape[0]] += self._fixed_b

    # marginalize
    if marginalize_state_size > 0:
      A00 = A[:marginalize_state_size, :marginalize_state_size]
      A10 = A[marginalize_state_size:, :marginalize_state_size]
      A11 = A[marginalize_state_size:, marginalize_state_size:]
      b0 = b[:marginalize_state_size]
      b1 = b[marginalize_state_size:]
      A = A11 - A10 @ npla.inv(A00) @ A10.T
      b = b1 - A10 @ npla.inv(A00) @ b0

    return A, b
