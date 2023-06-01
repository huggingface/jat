def collapse_none(x):
    """
    Recursively collapse list of None into None.

    Args:
        lst (Any): Any object. If it's a list, it will be recursively collapsed.

    Example:
        >>> collapse_none([1, 2])
        [1, 2]
        >>> collapse_none([1, 2, None])
        [1, 2, None]
        >>> collapse_none([1, 2, [None]])
        [1, 2, None]
        >>> collapse_none([1, 2, [None, None]])
        [1, 2, None]
        >>> collapse_none([1, 2, [None, 3]])
        [1, 2, [None, 3]]
        >>> collapse_none([1, 2, [None, [None]]])
        [1, 2, None]
        >>> collapse_none([None, None])
        None
    """
    if isinstance(x, list):
        x = [collapse_none(xx) for xx in x]
        if all(xx is None for xx in x):
            return None
    return x

if __name__ == "__main__":
    print(collapse_none(1))
    print(collapse_none([1, 2]))
    print(collapse_none([1, 2, None]))
    print(collapse_none([1, 2, [None]]))
    print(collapse_none([1, 2, [None, None]]))
    print(collapse_none([1, 2, [None, 3]]))
    print(collapse_none([1, 2, [None, [None]]]))
    print(collapse_none([None, None]))

