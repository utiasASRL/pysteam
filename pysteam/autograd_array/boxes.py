import numpy as np
from ..autograd.extend import Box, SparseObject, primitive, defvjp, vspace

class ArrayBox(Box):
    __slots__ = []

    @primitive
    def __getitem__(A, idx): return A[idx]

    # Constants w.r.t float data just pass though
    # shape = property(lambda self: self._value.shape)
    # ndim  = property(lambda self: self._value.ndim)
    # size  = property(lambda self: self._value.size)
    # dtype = property(lambda self: self._value.dtype)
    # def __len__(self): return len(self._value)
    def __hash__(self): return id(self)

ArrayBox.register(np.ndarray)
for type_ in [float, np.float64, np.float32, np.float16]:
    ArrayBox.register(type_)

@primitive
def untake(x, idx, vs):
    if isinstance(idx, list) and (len(idx) == 0 or not isinstance(idx[0], slice)):
        idx = np.array(idx, dtype='int64')
    def mut_add(A):
        np.add.at(A, idx, x)
        return A
    return SparseObject(vs, mut_add)
defvjp(ArrayBox.__getitem__, lambda ans, A, idx: lambda g: untake(g, idx, vspace(A)))
defvjp(untake, lambda ans, x, idx, _: lambda g: g[idx])