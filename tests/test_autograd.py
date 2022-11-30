import numpy as np
from pylgmath import Transformation, se3op

from pysteam.autograd.differential_operators import grad, jacobian
from pysteam.autograd_array import vjps as arrayprims
from pysteam.autograd_se3 import vjps as se3prims


def test_array():
    error = lambda x, y: arrayprims.multiply(
        arrayprims.multiply(arrayprims.multiply(x, y), y), y
    )
    jac = jacobian(error, (0, 1))(np.array([2.0]), np.array([3.0]))
    print(jac)


test_array()


def test_se3():
    error = lambda T1, T2: se3prims.log(se3prims.compose(T1, se3prims.inv(T2)))
    jac = jacobian(error, (0, 1))(Transformation(), Transformation())
    print(jac)


test_se3()
