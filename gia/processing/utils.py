def nested_like(x, val):
    if isinstance(x, list):
        return [nested_like(x_i, val) for x_i in x]
    else:
        return val
