import numpy as np
from pylgmath import se3op, Transformation
from pysteam import problem, solver
from pysteam.evaluatable import SE3StateVar, LogMapEvaluator, InverseEvaluator, ComposeEvaluator

np.set_printoptions(precision=2)


def test_identity():
  # construct random pose as initial guess
  xi_ab = np.random.uniform(-np.pi / 2, np.pi / 2, size=(6, 1))
  hat_T_ba = Transformation(xi_ab=xi_ab)

  # wrap as a variable
  hat_T_ba_var = SE3StateVar(hat_T_ba)

  # setup loss function, noise model, error_func and cost term
  loss_func = problem.L2LossFunc()
  noise_model = problem.StaticNoiseModel(np.eye(6), "covariance")
  error_func = LogMapEvaluator(hat_T_ba_var)

  cost_term = problem.WeightedLeastSquareCostTerm(error_func, noise_model, loss_func)

  # setup the optimization problem
  opt_prob = problem.OptimizationProblem()
  opt_prob.add_state_var(hat_T_ba_var)
  opt_prob.add_cost_term(cost_term)

  # construct the solver
  optimizer = solver.GaussNewtonSolver(opt_prob)

  # solve the problem
  optimizer.optimize()

  # check result
  assert np.allclose(hat_T_ba.matrix(), np.eye(4))


def test_direct_measure():
  # construct random pose as ground truth
  xi_ab = np.random.uniform(-np.pi / 2, np.pi / 2, size=(6, 1))
  meas_T_ba = Transformation(xi_ab=xi_ab)

  # apply random perturbation
  perturb = np.random.uniform(-np.pi / 2, np.pi / 2, size=(6, 1))
  hat_T_ba = Transformation(xi_ab=perturb + xi_ab)

  # wrap as a variable
  meas_T_ba_var = SE3StateVar(meas_T_ba, locked=True)
  hat_T_ba_var = SE3StateVar(hat_T_ba)

  # setup loss function, noise model, error_func and cost term
  loss_func = problem.L2LossFunc()
  noise_model = problem.StaticNoiseModel(np.eye(6), "covariance")
  error_func = LogMapEvaluator(ComposeEvaluator(meas_T_ba_var, InverseEvaluator(hat_T_ba_var)))

  cost_term = problem.WeightedLeastSquareCostTerm(error_func, noise_model, loss_func)

  # setup the optimization problem
  opt_prob = problem.OptimizationProblem()
  opt_prob.add_state_var(hat_T_ba_var)
  opt_prob.add_cost_term(cost_term)

  # construct the solver
  optimizer = solver.GaussNewtonSolver(opt_prob)

  # solve the problem
  optimizer.optimize()

  # check result
  assert np.allclose(hat_T_ba.matrix(), meas_T_ba.matrix())


def test_relative_change():
  # construct random pose as ground truth
  T_10 = Transformation(xi_ab=np.random.uniform(-np.pi / 2, np.pi / 2, size=(6, 1)))
  T_20 = Transformation(xi_ab=np.random.uniform(-np.pi / 2, np.pi / 2, size=(6, 1)))
  meas_T_21 = Transformation(T_ba=T_20.matrix() @ T_10.inverse().matrix())

  # apply perturbation
  perturb = se3op.vec2tran(np.random.uniform(-np.pi / 2, np.pi / 2, size=(2, 6, 1)))
  hat_T_10 = Transformation(T_ba=perturb[0] @ T_10.matrix())
  hat_T_20 = Transformation(T_ba=perturb[1] @ T_20.matrix())

  # wrap as a variable
  hat_T_10_var = SE3StateVar(hat_T_10)
  hat_T_20_var = SE3StateVar(hat_T_20)
  meas_T_10_var = SE3StateVar(T_10, locked=True)
  meas_T_21_var = SE3StateVar(meas_T_21, locked=True)

  # setup loss function, noise model, error_func and cost term
  loss_func = problem.L2LossFunc()
  noise_model = problem.StaticNoiseModel(np.eye(6), "covariance")
  # prior term on the first pose
  error_func1 = LogMapEvaluator(ComposeEvaluator(meas_T_10_var, InverseEvaluator(hat_T_10_var)))
  cost_term1 = problem.WeightedLeastSquareCostTerm(error_func1, noise_model, loss_func)
  # measured relative change
  error_func2 = LogMapEvaluator(
      ComposeEvaluator(ComposeEvaluator(meas_T_21_var, hat_T_10_var), InverseEvaluator(hat_T_20_var)))
  cost_term2 = problem.WeightedLeastSquareCostTerm(error_func2, noise_model, loss_func)

  # setup the optimization problem
  opt_prob = problem.OptimizationProblem()
  opt_prob.add_state_var(hat_T_10_var, hat_T_20_var)
  opt_prob.add_cost_term(cost_term1, cost_term2)

  # construct the solver
  optimizer = solver.GaussNewtonSolver(opt_prob)

  # solve the problem
  optimizer.optimize()

  # check result
  assert np.allclose(hat_T_10.matrix(), T_10.matrix())
  assert np.allclose(hat_T_20.matrix(), T_20.matrix())
