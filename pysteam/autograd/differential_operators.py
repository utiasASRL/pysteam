"""Convenience functions built on top of `make_vjp`."""
from .wrap_util import unary_to_nary
from .core import make_vjp as _make_vjp
from .extend import primitive, defvjp_argnum, vspace
from .builtins import reshape, stack

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

@unary_to_nary
def jacobian(fun, x):
    """
    Returns a function which computes the Jacobian of `fun` with respect to
    positional argument number `argnum`, which must be a scalar or array. Unlike
    `grad` it is not restricted to scalar-output functions, but also it cannot
    take derivatives with respect to some argument types (like lists or dicts).
    If the input to `fun` has shape (in1, in2, ...) and the output has shape
    (out1, out2, ...) then the Jacobian has shape (out1, out2, ..., in1, in2, ...).
    """
    vjp, ans = _make_vjp(fun, x)
    ans_vspace = vspace(ans)
    jacobian_shape = ans_vspace.shape + vspace(x).shape
    grads = map(vjp, ans_vspace.standard_basis())
    return reshape(stack(grads), jacobian_shape)

@unary_to_nary
def hessian(fun, x):
    "Returns a function that computes the exact Hessian."
    return jacobian(jacobian(fun))(x)

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

def checkpoint(fun):
    """Returns a checkpointed version of `fun`, where intermediate values
    computed during the forward pass of `fun` are discarded and then recomputed
    for the backward pass. Useful to save memory, effectively trading off time
    and memory. See e.g. arxiv.org/abs/1604.06174.
    """
    def wrapped_grad(argnum, ans, args, kwargs):
        return make_vjp(fun, argnum)(*args, **kwargs)[0]
    wrapped = primitive(fun)
    defvjp_argnum(wrapped, wrapped_grad)
    return wrapped
