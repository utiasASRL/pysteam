import numpy as np


def getQ(dt: float, ad: np.ndarray, qcd: np.ndarray = np.ones(6)):
  # constants
  adinv = 1.0 / ad

  adt = ad * dt
  adt2 = adt * adt
  adt3 = adt2 * adt

  expon = np.exp(-adt)
  expon2 = np.exp(-2 * adt)

  Q11 = np.diag(qcd * (adinv**4) * (1 - expon2 + 2 * adt + 2. / 3. * adt3 - 2 * adt2 - 4 * adt * expon))
  Q12 = np.diag(qcd * (adinv**3) * (expon2 + 1 - 2 * expon + 2 * adt * expon - 2 * adt + adt2))
  Q13 = np.diag(qcd * (adinv**2) * (1 - expon2 - 2 * adt * expon))
  Q22 = np.diag(qcd * (adinv**2) * (4 * expon - 3 - expon2 + 2 * adt))
  Q23 = np.diag(qcd * adinv * (expon2 + 1 - 2 * expon))
  Q33 = np.diag(qcd * (1 - expon2))
  Q = np.block([[Q11, Q12, Q13], [Q12.T, Q22, Q23], [Q13.T, Q23.T, Q33]])
  return Q


def getTran(dt: float, ad: np.ndarray):
  adinv = 1.0 / ad
  adt = ad * dt
  expon = np.exp(-adt)

  Q11 = Q22 = np.eye(6)
  Q21 = Q31 = Q32 = np.zeros((6, 6))
  Q12 = dt * np.eye(6)
  Q13 = np.diag((adt - 1 + expon) * adinv * adinv)  # TODO: double check this
  Q23 = np.diag((1 - expon) * adinv)
  Q33 = np.diag(expon)

  Tran = np.block([[Q11, Q12, Q13], [Q21, Q22, Q23], [Q31, Q32, Q33]])
  return Tran
