from functools import reduce

def compose2(f, g):
    return lambda *args, **kwargs: f(g(*args, **kwargs))

def compose(*fs):
    """compose functions

    >>> f = compose(abs, min)
    >>> f([1, -2, 3])
    2
    """
    return reduce(compose2, fs)

def kw(f, **kwargs):
    """pass keyword arguments in advance
    
    >>> desc = kw(sorted, reverse=True)
    >>> desc([4, 1, 3, 2])
    [4, 3, 2, 1]
    """
    return lambda *args: f(*args, **kwargs)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
