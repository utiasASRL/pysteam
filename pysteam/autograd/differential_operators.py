"""Convenience functions built on top of `make_vjp`."""
from itertools import repeat, starmap
from .wrap_util import unary_to_nary
from .core import make_vjp as _make_vjp
from .extend import primitive, defvjp_argnum, vspace

make_vjp = unary_to_nary(_make_vjp)

@unary_to_nary
def grad(fun, x):
    """
    Returns a function which computes the gradient of `fun` with respect to
    positional argument number `argnum`. The returned function takes the same
    arguments as `fun`, but returns the gradient instead. The function `fun`
    should be scalar-valued. The gradient has the same type as the argument."""
    vjp, ans = _make_vjp(fun, x)
    if not vspace(ans).size == 1:
        raise TypeError("Grad only applies to real scalar-output functions. "
                        "Try jacobian, elementwise_grad or holomorphic_grad.")
    return vjp(vspace(ans).ones())

@unary_to_nary
def value_and_grad(fun, x):
    """Returns a function that returns both value and gradient. Suitable for use
    in scipy.optimize"""
    vjp, ans = _make_vjp(fun, x)
    if not vspace(ans).size == 1:
        raise TypeError("value_and_grad only applies to real scalar-output "
                        "functions. Try jacobian, elementwise_grad or "
                        "holomorphic_grad.")
    return ans, vjp(vspace(ans).ones())

@unary_to_nary
def elementwise_grad(fun, x):
    """
    Returns a function that computes the sum of each column of the Jacobian of
    `fun`, in one pass. If the Jacobian is diagonal, then this is the diagonal
    of the Jacobian.
    """
    vjp, ans = _make_vjp(fun, x)
    if vspace(ans).iscomplex:
        raise TypeError("Elementwise_grad only applies to real-output functions.")
    return vjp(vspace(ans).ones())


import numpy as np

@unary_to_nary
def jacobian(fun, x):
    """
    Note that this is not the same as how AutoGrad computes the Jacobian. It
    accepts multiple variables but cannot be used to compute Hessian because
    np.stack and np.reshape are not autograd friendly.
    """
    vjp, ans = _make_vjp(fun, x)
    ans_vspace = vspace(ans)
    in_vspace_shape = list(vspace(x).shape)
    jacobian_shape = starmap(lambda a, b: a+b.shape,
                             zip(repeat(ans_vspace.shape, len(in_vspace_shape)), in_vspace_shape))
    grads = map(vjp, ans_vspace.standard_basis())
    # following operations are not autograd-friendly
    stacked = map(np.stack, zip(*grads))
    jacs = starmap(np.reshape, zip(stacked, jacobian_shape))
    return tuple(jacs)
