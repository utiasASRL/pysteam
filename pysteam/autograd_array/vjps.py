from ..autograd.extend import primitive, defvjp

@primitive
def add(x, y):
    return x + y
defvjp(add, lambda ans, x, y : lambda g: g,
            lambda ans, x, y : lambda g: g)

@primitive
def multiply(x, y):
    return x * y
defvjp(multiply, lambda ans, x, y : lambda g: multiply(y, g),
                 lambda ans, x, y : lambda g: multiply(x, g))