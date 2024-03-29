from functools import reduce
from itertools import count

from .tracer import Box, Node, getval, isbox, primitive, toposort, trace

# -------------------- reverse mode --------------------


def make_vjp(fun, x):
    start_node = VJPNode.new_root()
    end_value, end_node = trace(start_node, fun, x)
    if end_node is None:

        def vjp(g):
            return vspace(x).zeros()

    else:

        def vjp(g):
            return backward_pass(g, end_node)

    return vjp, end_value


def backward_pass(g, end_node):
    outgrads = {end_node: (g, False)}
    for node in toposort(end_node):
        outgrad = outgrads.pop(node)
        ingrads = node.vjp(outgrad[0])
        for parent, ingrad in zip(node.parents, ingrads):
            outgrads[parent] = add_outgrads(outgrads.get(parent), ingrad)
    return outgrad[0]


class VJPNode(Node):
    __slots__ = ["parents", "vjp"]

    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.parents = parents
        try:
            vjpmaker = primitive_vjps[fun]
        except KeyError:
            fun_name = getattr(fun, "__name__", fun)
            raise NotImplementedError(
                "VJP of {} wrt argnums {} not defined".format(fun_name, parent_argnums)
            )
        self.vjp = vjpmaker(parent_argnums, value, args, kwargs)

    def initialize_root(self):
        self.parents = []
        self.vjp = lambda g: ()


primitive_vjps = {}


def defvjp_argnums(fun, vjpmaker):
    primitive_vjps[fun] = vjpmaker


def defvjp_argnum(fun, vjpmaker):
    def vjp_argnums(argnums, *args):
        vjps = [vjpmaker(argnum, *args) for argnum in argnums]
        return lambda g: (vjp(g) for vjp in vjps)

    defvjp_argnums(fun, vjp_argnums)


def defvjp(fun, *vjpmakers, **kwargs):
    argnums = kwargs.get("argnums", count())
    vjps_dict = {
        argnum: translate_vjp(vjpmaker, fun, argnum)
        for argnum, vjpmaker in zip(argnums, vjpmakers)
    }

    def vjp_argnums(argnums, ans, args, kwargs):
        L = len(argnums)
        # These first two cases are just optimizations
        if L == 1:
            argnum = argnums[0]
            try:
                vjpfun = vjps_dict[argnum]
            except KeyError:
                raise NotImplementedError(
                    "VJP of {} wrt argnum 0 not defined".format(fun.__name__)
                )
            vjp = vjpfun(ans, *args, **kwargs)
            return lambda g: (vjp(g),)
        elif L == 2:
            argnum_0, argnum_1 = argnums
            try:
                vjp_0_fun = vjps_dict[argnum_0]
                vjp_1_fun = vjps_dict[argnum_1]
            except KeyError:
                raise NotImplementedError(
                    "VJP of {} wrt argnums 0, 1 not defined".format(fun.__name__)
                )
            vjp_0 = vjp_0_fun(ans, *args, **kwargs)
            vjp_1 = vjp_1_fun(ans, *args, **kwargs)
            return lambda g: (vjp_0(g), vjp_1(g))
        else:
            vjps = [vjps_dict[argnum](ans, *args, **kwargs) for argnum in argnums]
            return lambda g: (vjp(g) for vjp in vjps)

    defvjp_argnums(fun, vjp_argnums)


def translate_vjp(vjpfun, fun, argnum):
    if vjpfun is None:
        return lambda ans, *args, **kwargs: lambda g: vspace(args[argnum]).zeros()
    elif callable(vjpfun):
        return vjpfun
    else:
        raise Exception("Bad VJP '{}' for '{}'".format(vjpfun, fun.__name__))


# -------------------- vector behavior --------------------


def add_outgrads(prev_g_flagged, g):
    sparse = type(g) in sparse_object_types
    if prev_g_flagged:
        vs = vspace(g)
        prev_g, mutable = prev_g_flagged
        if mutable:
            if sparse:
                return sparse_add(vs, prev_g, g), True
            else:
                return vs.mut_add(prev_g, g), True
        else:
            if sparse:
                prev_g_mutable = vs.mut_add(None, prev_g)
                return sparse_add(vs, prev_g_mutable, g), True
            else:
                return vs.add(prev_g, g), True
    else:
        if sparse:
            return sparse_add(vspace(g), None, g), True
        else:
            return g, False


def sum_outgrads(gs):
    return reduce(add_outgrads, gs, None)[0]


@primitive
def sparse_add(vs, x_prev, x_new):
    x_prev = x_prev if x_prev is not None else vs.zeros()
    return x_new.mut_add(x_prev)


class VSpace(object):
    __slots__ = []
    mappings = {}
    iscomplex = False

    def __init__(self, value):
        pass

    def zeros(self):
        assert False, repr(self)

    def ones(self):
        assert False, repr(self)

    def standard_basis(self):
        assert False, repr(self)

    def randn(self):
        assert False, repr(self)

    @primitive
    def mut_add(self, x_prev, x_new):
        x_prev = x_prev if x_prev is not None else self.zeros()
        return self._mut_add(x_prev, x_new)

    @primitive
    def add(self, x_prev, x_new):
        return self._add(x_prev, x_new)

    @primitive
    def scalar_mul(self, x, a):
        return self._scalar_mul(x, a)

    @primitive
    def inner_prod(self, x, y):
        return self._inner_prod(x, y)

    @primitive
    def covector(self, x):
        return self._covector(x)

    def _add(self, x, y):
        return x + y

    def _mut_add(self, x, y):
        x += y
        return x

    def _scalar_mul(self, x, a):
        return x * a

    def _inner_prod(self, x, y):
        assert False

    def _covector(self, x):
        return x

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

    def __repr__(self):
        return "{}_{}".format(type(self).__name__, self.__dict__)

    @classmethod
    def register(cls, value_type, vspace_maker=None):
        if vspace_maker:
            VSpace.mappings[value_type] = vspace_maker
        else:
            VSpace.mappings[value_type] = cls


def vspace(value):
    try:
        return VSpace.mappings[type(value)](value)
    except KeyError:
        if isbox(value):
            return vspace(getval(value))
        else:
            raise TypeError(
                "Can't find vector space for value {} of type {}. "
                "Valid types are {}".format(value, type(value), VSpace.mappings.keys())
            )


class SparseBox(Box):
    __slots__ = []


class SparseObject(object):
    __slots__ = ["vs", "mut_add"]

    def __init__(self, vs, mut_add):
        self.vs = vs
        self.mut_add = mut_add


VSpace.register(SparseObject, lambda x: x.vs)
SparseBox.register(SparseObject)
sparse_object_types = {SparseObject, SparseBox}

# -------------------- core reverse mode grads --------------------

identity_vjp = lambda argnums, *args: lambda g: g
defvjp(sparse_add, None, identity_vjp, identity_vjp)
defvjp(VSpace.add, None, identity_vjp, identity_vjp)
defvjp(VSpace.mut_add, None, identity_vjp, identity_vjp)
defvjp(
    VSpace.inner_prod,
    None,
    lambda ans, vs, x, y: lambda g: vs.covector(vs.scalar_mul(y, g)),
    lambda ans, vs, x, y: lambda g: vs.covector(vs.scalar_mul(x, g)),
)
defvjp(VSpace.covector, None, lambda ans, vs, x: lambda g: vs.covector(g))
defvjp(
    VSpace.scalar_mul,
    None,
    lambda ans, vs, x, a: lambda g: vs.covector(vs.scalar_mul(vs.covector(g), a)),
    lambda ans, vs, x, a: lambda g: vs.inner_prod(g, vs.covector(x)),
)
