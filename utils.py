def to_2tuple(x):
    """
    Return x as a 2-tuple; if x is scalar, return (x, x).

    Args:
        x: Scalar or sequence value.
    """
    return x if isinstance(x, (list, tuple)) else (x, x)
