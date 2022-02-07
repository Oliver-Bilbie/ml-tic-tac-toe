"""
The data service module contains the logic for feature engineering and data manipulation
"""

import numpy as np
import pandas as pd

from src.service import generators


def onehot_encode(dataset):
    """Onehot encodes the board state features

    Args:
        dataset [DataFrame]: Pandas DataFrame containing the dataset to be onehot encoded

    Returns:
        [DataFrame]: Onehot encoded dataset
    """

    columns = generators.get_onehot_column_names()
    dataset = pd.get_dummies(dataset.iloc[:, :9])[columns]

    return dataset


def balance_dataset(dataset, tolerance=0.05):
    """Balances a dataset by removing random entries.
    Accepts an optional tolerance parameter which corresponds to a maxiumum proportional difference
    between the frequency of outcomes (Defaults to 5%).

    Args:
        dataset [DataFrame] : DataFrame containing the dataset
        tolerance [Float]   : Maximum allowed difference in outcome frequency

    Returns:
        [DataFrame]: DataFrame containing a balanced dataset
    """

    while not check_balancing(dataset.copy(), tolerance):
        # Randomly select a row until this row corresponds to an over-represented outcome
        row = np.random.randint(0, dataset.shape[0])
        result = dataset.index[0]
        if check_overrepresented(dataset.copy(), result, tolerance):
            index = dataset.index[row]
            dataset.drop(index=index, inplace=True)

    return dataset


def check_balancing(dataset, tolerance):
    """Checks whether a dataset is balanced. Returns "True" if balanced
    (within given tolerance), otherwise "False".

    Args:
        dataset [DataFrame] : DataFrame containing the dataset
        tolerance [Float]   : Maximum allowed difference in outcome frequency

    Returns:
        [Boolean]: True if balanced, otherwise False
    """

    balanced = True
    outcomes = dataset.index.value_counts()
    count = outcomes.iloc[0]

    for outcome in outcomes[1:]:
        if outcome > count * (1 + tolerance) or outcome < count * (1 - tolerance):
            balanced = False
            break

    return balanced


def check_overrepresented(dataset, result, tolerance):
    """Checks whether an outcome is over-represented in a given dataset.
    Returns "True" if over-represented (within given tolerance), otherwise "False".

    Args:
        dataset [DataFrame] : DataFrame containing the dataset
        tolerance [Float]   : Maximum allowed difference in outcome frequency

    Returns:
        [Boolean]: True if balanced, otherwise False
    """

    overrepresented = False
    outcomes = dataset.index.value_counts()
    count = outcomes.loc[result]

    for outcome in outcomes:
        if outcome < count * (1 - tolerance):
            overrepresented = True
            break

    return overrepresented


def calculate_move_counts(dataset):
    """Adds additional features to a dataset corresponding to the number of
    Xs and Os on the game board. These features are named "x_count" and "o_count"
    respectively.

    Args:
        dataset [DataFrame] : DataFrame containing the dataset

    Returns:
        [DataFrame]: DataFrame containing the dataset with additional features

    """

    # Replace all X values with 1 and all other values with 0
    x_df = dataset.iloc[:, 1:].replace(["x", "o", "b"], [1, 0, 0])
    # Sum each row
    x_count = x_df.sum(axis=1)

    # Replace all O values with 1 and all other values with 0
    o_df = dataset.iloc[:, 1:].replace(["x", "o", "b"], [0, 1, 0])
    # Sum each row
    o_count = o_df.sum(axis=1)

    # Add features to dataset
    dataset["x_count"] = x_count
    dataset["o_count"] = o_count

    return dataset


def calculate_adjacent_symbols(dataset):
    """Adds additional features to a dataset corresponding to the number of
    Xs or Os in adjacent squares. This is broken down further based on whether
    the axis in which they are adjacent is horizontal, vertical, diagonal (positive),
    or diagonal (negative). The convention used for diagonal naming is based on
    positive and negative gradient.

    The new features are called:
    [x_adj_horizontal], [x_adj_vertical], [x_adj_diagonal_pos], [x_adj_diagonal_neg],
    [o_adj_horizontal], [o_adj_vertical], [o_adj_diagonal_pos], [o_adj_diagonal_neg]

    Args:
        dataset [DataFrame] : DataFrame containing the dataset

    Returns:
        [DataFrame]: DataFrame containing the dataset with additional features

    """

    v_positions = ["top", "middle", "bottom"]  # All possible vertical positions
    h_positions = ["left", "middle", "right"]  # All possible horizontal positions

    for player in ["x", "o"]:
        # evaluate adj_horizontal
        dataset[f"{player}_adj_horizontal"] = 0
        for v_position in v_positions:
            for h_index in [0, 1]:
                adjacent_rows = (
                    dataset[f"{v_position}-{h_positions[h_index]}-square"] == player
                ) & (dataset[f"{v_position}-{h_positions[h_index+1]}-square"] == player)

                dataset.loc[adjacent_rows, f"{player}_adj_horizontal"] += 1

        # evaluate adj_vertical
        dataset[f"{player}_adj_vertical"] = 0
        for v_index in [0, 1]:
            for h_position in h_positions:
                adjacent_rows = (
                    dataset[f"{v_positions[v_index]}-{h_position}-square"] == player
                ) & (dataset[f"{v_positions[v_index+1]}-{h_position}-square"] == player)

                dataset.loc[adjacent_rows, f"{player}_adj_vertical"] += 1

        # evaluate adj_diagonal_pos
        dataset[f"{player}_adj_diagonal_pos"] = 0
        for v_index in [0, 1]:
            for h_index in [0, 1]:
                adjacent_rows = (
                    dataset[f"{v_positions[v_index+1]}-{h_positions[h_index]}-square"]
                    == player
                ) & (
                    dataset[f"{v_positions[v_index]}-{h_positions[h_index+1]}-square"]
                    == player
                )

                dataset.loc[adjacent_rows, f"{player}_adj_diagonal_pos"] += 1

        # evaluate adj_diagonal_neg
        dataset[f"{player}_adj_diagonal_neg"] = 0
        for v_index in [0, 1]:
            for h_index in [0, 1]:
                adjacent_rows = (
                    dataset[f"{v_positions[v_index]}-{h_positions[h_index]}-square"]
                    == player
                ) & (
                    dataset[f"{v_positions[v_index+1]}-{h_positions[h_index+1]}-square"]
                    == player
                )

                dataset.loc[adjacent_rows, f"{player}_adj_diagonal_neg"] += 1

    return dataset
