def get_onehot_column_names():
    """Returns a set of column names corresponding to a onehot encoded board state.

    Returns:
        [String[]]: List of column names
    """

    v_positions = ["top", "middle", "bottom"]  # All possible vertical positions
    h_positions = ["left", "middle", "right"]  # All possible horizontal positions
    players = ["x", "o", "b"]  # All possible players (including blanks)
    column_names = []

    for p in players:
        for y in v_positions:
            for x in h_positions:
                column_names.append(f"{y}-{x}-square_{p}")

    return column_names


def get_param_grid():
    """Returns a range of parameters used to optimize the model.
    Factored out for ease of testing.

    Returns:
        param_grid: Dictionary containing parameter range"""

    param_grid = {
        "n_estimators": [50, 100, 250],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [4, 8, 16, 32, 64, 128, 256],
        "criterion": ["gini", "entropy"],
    }

    # Uncomment for testing
    # param_grid = {
    #     "n_estimators": [30],
    #     "max_features": ["sqrt"],
    #     "max_depth": [2],
    #     "criterion": ["gini"],
    # }

    return param_grid
