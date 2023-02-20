
def convert_to_string(state):
    """Convert a state to a string."""
    return ",".join([str(s) for s in state])


def convert_to_list(state_str):
    """Convert a state string to a list."""
    return [int(s) for s in state_str.split(",")]


def decay_param(param: float,
                 decay: float,
                 min_param: float) -> float:
    """Decay a parameter.

    Args:
        param: parameter to decay.
        decay: decay rate.
        min_param: minimum value of the parameter.

    Returns:
        param: updated parameter.
    """
    return max(param * (1 - decay), min_param)