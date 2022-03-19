import numpy as np

from pylgmath import se3op, Transformation

from ..autograd.extend import primitive, defvjp

@primitive
def pass_through(T: Transformation) -> Transformation:
    return T
defvjp(pass_through, lambda ans, T: lambda g: g)


@primitive
def log(T: Transformation) -> Transformation:
    return T.vec().squeeze()
def grad_log(ans: np.ndarray, T: Transformation):
    def grad_log_(g: np.ndarray):
        res = g @ se3op.vec2jacinv(ans[..., None])
        return res
    return grad_log_
defvjp(log, grad_log)


@primitive
def inv(T: Transformation) -> Transformation:
    return T.inverse()
def grad_inv(ans: Transformation, T: Transformation):
    def grad_inv_(g: np.ndarray):
        return -g @ T.adjoint()
    return grad_inv_
defvjp(inv, grad_inv)


@primitive
def compose(T1: Transformation, T2: Transformation) -> Transformation:
    return T1 @ T2
def grad_compose_T1(ans: Transformation, T1: Transformation, T2: Transformation):
    def grad_compose_T1_(g: np.ndarray):
        return g
    return grad_compose_T1_
def grad_compose_T2(ans: Transformation, T1: Transformation, T2: Transformation):
    def grad_compose_T2_(g: np.ndarray):
        return g @ T1.adjoint()
    return grad_compose_T2_
defvjp(compose, grad_compose_T1, grad_compose_T2)