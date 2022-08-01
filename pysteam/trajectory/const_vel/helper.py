import numpy as np


def getQinv(dt: float, qcd: np.ndarray = None):
  dtinv = 1.0 / dt

  qinv00 = 12.0 * (dtinv**3)
  qinv11 = 4.0 * (dtinv)
  qinv01 = qinv10 = (-6.0) * (dtinv**2)

  if qcd is None:
    return np.array([
        [qinv00, qinv01],
        [qinv10, qinv11],
    ])

  Qcinv = np.diag(1.0 / qcd)
  return np.block([
      [qinv00 * Qcinv, qinv01 * Qcinv],
      [qinv10 * Qcinv, qinv11 * Qcinv],
  ])


def getQ(dt: float):

  q00 = (dt**3) / 3.0
  q11 = dt
  q01 = q10 = (dt**2) / 2.0

  return np.array([
      [q00, q01],
      [q10, q11],
  ])


def getTran(dt: float):
  Tran = np.array([
      [1.0, dt],
      [0.0, 1.0],
  ])
  return Tran
