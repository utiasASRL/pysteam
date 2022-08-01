import numpy as np


def getQ(dt: float, qcd: np.ndarray = np.ones(6)):
  Q11 = np.diag(qcd * (dt**5) / 20.0)
  Q12 = np.diag(qcd * (dt**4) / 8.0)
  Q13 = np.diag(qcd * (dt**3) / 6.0)
  Q22 = np.diag(qcd * (dt**3) / 3.0)
  Q23 = np.diag(qcd * (dt**2) / 2.0)
  Q33 = np.diag(qcd * dt)
  Q = np.block([[Q11, Q12, Q13], [Q12.T, Q22, Q23], [Q13.T, Q23.T, Q33]])
  return Q


def getTran(dt: float):
  Q11 = Q22 = Q33 = np.eye(6)
  Q21 = Q31 = Q32 = np.zeros((6, 6))
  Q12 = Q23 = dt * np.eye(6)
  Q13 = 0.5 * (dt**2) * np.eye(6)
  Tran = np.block([[Q11, Q12, Q13], [Q21, Q22, Q23], [Q31, Q32, Q33]])
  return Tran
