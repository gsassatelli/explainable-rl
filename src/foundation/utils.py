def convert_to_string(state):
    """Convert a state to a string.

    Args:
        state (list): State to convert.

    Returns:
        str: State as a string.
    """
    return ",".join([str(s) for s in state])


def convert_to_list(state_str):
    """Convert a state string to a list.

    Args:
        state_str (str): State as a string.

    Returns:
        list: State as a list.
    """
    return [int(s) for s in state_str.split(",")]


def decay_param(param,
                decay,
                min_param):
    """Decay a parameter.

    Args:
        param (float): Parameter to decay.
        decay (float): Decay rate.
        min_param (float): Minimum value of the parameter.

    Returns:
        float: Updated parameter.
    """
    return max(param * (1 - decay), min_param)
