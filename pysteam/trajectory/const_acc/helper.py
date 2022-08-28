import numpy as np


def getQinv(dt: float, qcd: np.ndarray = None):
  dtinv = 1.0 / dt

  qinv00 = 720.0 * (dtinv**5)
  qinv11 = 192.0 * (dtinv**3)
  qinv22 = 9.0 * (dtinv)
  qinv01 = qinv10 = (-360.0) * (dtinv**4)
  qinv02 = qinv20 = (60.0) * (dtinv**3)
  qinv12 = qinv21 = (-36.0) * (dtinv**2)

  if qcd is None:
    return np.array([
        [qinv00, qinv01, qinv02],
        [qinv10, qinv11, qinv12],
        [qinv20, qinv21, qinv22],
    ])

  Qcinv = np.diag(1.0 / qcd)
  return np.block([
      [qinv00 * Qcinv, qinv01 * Qcinv, qinv02 * Qcinv],
      [qinv10 * Qcinv, qinv11 * Qcinv, qinv12 * Qcinv],
      [qinv20 * Qcinv, qinv21 * Qcinv, qinv22 * Qcinv],
  ])


def getQ(dt: float):
  q00 = (dt**5) / 20.0
  q11 = (dt**3) / 3.0
  q22 = dt
  q01 = q10 = (dt**4) / 8.0
  q02 = q20 = (dt**3) / 6.0
  q12 = q21 = (dt**2) / 2.0

  return np.array([
      [q00, q01, q02],
      [q10, q11, q12],
      [q20, q21, q22],
  ])


def getTran(dt: float):
  Tran = np.array([
      [1.0, dt, 0.5 * (dt**2)],
      [0.0, 1.0, dt],
      [0.0, 0.0, 1.0],
  ])
  return Tran
