import numpy as np


def getQ(dt: float, add: np.ndarray, qcd: np.ndarray = np.ones(6)):
    dim = add.squeeze().shape[0]
    assert dim == qcd.squeeze().shape[0]

    Q11 = np.zeros((dim, dim))
    Q12 = np.zeros((dim, dim))
    Q13 = np.zeros((dim, dim))
    Q22 = np.zeros((dim, dim))
    Q23 = np.zeros((dim, dim))
    Q33 = np.zeros((dim, dim))

    for i in range(dim):
        ad = add.squeeze()[i]
        qc = qcd.squeeze()[i]

        if np.abs(ad) >= 1.0:
            adi = 1.0 / ad
            adi2 = adi * adi
            adi3 = adi * adi2
            adi4 = adi * adi3
            adi5 = adi * adi4
            adt = ad * dt
            adt2 = adt * adt
            adt3 = adt2 * adt
            expon = np.exp(-adt)
            expon2 = np.exp(-2 * adt)
            Q11[i, i] = qc * (
                0.5
                * adi5
                * (
                    1
                    - expon2
                    + 2 * adt
                    + (2.0 / 3.0) * adt3
                    - 2 * adt2
                    - 4 * adt * expon
                )
            )
            Q12[i, i] = qc * (
                0.5 * adi4 * (expon2 + 1 - 2 * expon + 2 * adt * expon - 2 * adt + adt2)
            )
            Q13[i, i] = qc * 0.5 * adi3 * (1 - expon2 - 2 * adt * expon)
            Q22[i, i] = qc * 0.5 * adi3 * (4 * expon - 3 - expon2 + 2 * adt)
            Q23[i, i] = qc * 0.5 * adi2 * (expon2 + 1 - 2 * expon)
            Q33[i, i] = qc * 0.5 * adi * (1 - expon2)
        else:
            dt2 = dt * dt
            dt3 = dt * dt2
            dt4 = dt * dt3
            dt5 = dt * dt4
            dt6 = dt * dt5
            dt7 = dt * dt6
            dt8 = dt * dt7
            dt9 = dt * dt8
            ad2 = ad * ad
            ad3 = ad * ad2
            ad4 = ad * ad3
            # use Taylor series expansion about ad = 0
            Q11[i, i] = qc * (
                0.05 * dt5
                - 0.0277778 * dt6 * ad
                + 0.00992063 * dt7 * ad2
                - 0.00277778 * dt8 * ad3
                + 0.00065586 * dt9 * ad4
            )
            Q12[i, i] = qc * (
                0.125 * dt**4
                - 0.0833333 * dt5 * ad
                + 0.0347222 * dt6 * ad2
                - 0.0111111 * dt7 * ad3
                + 0.00295139 * dt8 * ad4
            )
            Q13[i, i] = qc * (
                (1 / 6) * dt3
                - (1 / 6) * dt4 * ad
                + 0.0916667 * dt5 * ad2
                - 0.0361111 * dt6 * ad3
                + 0.0113095 * dt7 * ad4
            )
            Q22[i, i] = qc * (
                (1 / 3) * dt3
                - 0.25 * dt4 * ad
                + 0.116667 * dt5 * ad2
                - 0.0416667 * dt6 * ad3
                + 0.0123016 * dt7 * ad4
            )
            Q23[i, i] = qc * (
                0.5 * dt2
                - 0.5 * dt3 * ad
                + 0.291667 * dt4 * ad2
                - 0.125 * dt5 * ad3
                + 0.0430556 * dt6 * ad4
            )
            Q33[i, i] = qc * (
                dt
                - dt2 * ad
                + (2 / 3) * dt3 * ad2
                - (1 / 3) * dt4 * ad3
                + 0.133333 * dt5 * ad4
            )

    return np.block(
        [
            [Q11, Q12, Q13],
            [Q12, Q22, Q23],
            [Q13, Q23, Q33],
        ]
    )


def getTran(dt: float, add: np.ndarray):
    dim = add.squeeze().shape[0]

    C1 = np.zeros((dim, dim))
    C2 = np.zeros((dim, dim))
    C3 = np.zeros((dim, dim))

    for i in range(dim):
        ad = add.squeeze()[i]
        if np.abs(ad) >= 1.0:
            adinv = 1.0 / ad
            adt = ad * dt
            expon: float = np.exp(-adt)
            C1[i, i] = (adt - 1.0 + expon) * adinv * adinv
            C2[i, i] = (1.0 - expon) * adinv
            C3[i, i] = expon
        else:
            C1[i, i] = (
                0.5 * dt**2
                - (1 / 6) * dt**3 * ad
                + (1 / 24) * dt**4 * ad**2
                - (1 / 120) * dt**5 * ad**3
                + (1 / 720) * dt**6 * ad**4
            )
            C2[i, i] = (
                dt
                - 0.5 * dt**2 * ad
                + (1 / 6) * dt**3 * ad**2
                - (1 / 24) * dt**4 * ad**3
                + (1 / 120) * dt**5 * ad**4
            )
            C3[i, i] = (
                1
                - dt * ad
                + 0.5 * dt**2 * ad**2
                - (1 / 6) * dt**3 * ad**3
                + (1 / 24) * dt**4 * ad**4
            )
    I = np.eye(dim)
    O = np.zeros((dim, dim))
    return np.block(
        [
            [I, dt * I, C1],
            [O, I, C2],
            [O, O, C3],
        ]
    )
