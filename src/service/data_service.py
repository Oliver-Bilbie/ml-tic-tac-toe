import pandas as pd
import numpy as np


def balance_dataset(dataset, tolerance=0.05):
    """Balances a dataset by removing random entries.
    Accepts an optional tolerance parameter which corresponds to a maxiumum proportional difference
    between the frequency of outcomes. Defaults to 5%.

    Args:
        dataset [DataFrame] : DataFrame containing the dataset
        tolerance [Float]   : Maximum allowed difference in outcome frequency

    Returns:
        [DataFrame]: DataFrame containing a balanced dataset
    """

    while not check_balancing(dataset.copy(), tolerance):
        # Randomly select a row until this row corresponds to an over-represented outcome
        row = np.random.randint(0, dataset.shape[0])
        result = dataset.iloc[row, 0]
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
    outcomes = dataset.iloc[:, 0].value_counts()
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
    outcomes = dataset.iloc[:, 0].value_counts()
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

    # Assign a prime value to all Xs
    x_df = dataset.iloc[:, 1:].replace(["x", "o", "b"], [1, 0, 0])
    x_df.loc[:, "top-left-square"] *= 2
    x_df.loc[:, "top-middle-square"] *= 3
    x_df.loc[:, "top-right-square"] *= 5
    x_df.loc[:, "middle-left-square"] *= 7
    x_df.loc[:, "middle-middle-square"] *= 11
    x_df.loc[:, "middle-right-square"] *= 13
    x_df.loc[:, "bottom-left-square"] *= 17
    x_df.loc[:, "bottom-middle-square"] *= 19
    x_df.loc[:, "bottom-right-square"] *= 23

    # Find the product of each board state
    products = (
        x_df.loc[:, "top-left-square"].values
        * x_df.loc[:, "top-middle-square"].values
        * x_df.loc[:, "top-right-square"].values
        * x_df.loc[:, "middle-left-square"].values
        * x_df.loc[:, "middle-middle-square"].values
        * x_df.loc[:, "middle-right-square"].values
        * x_df.loc[:, "bottom-left-square"].values
        * x_df.loc[:, "bottom-middle-square"].values
        * x_df.loc[:, "bottom-right-square"].values
    )

    # Take mods to find adjacent pairs
    x_adj_horizontal = np.zeros([dataset.shape[0]])

    test_values = products[products != 0] % 6

    test_values[test_values != 0] = -1
    test_values += 1

    print(test_values.sum())

    return dataset


with open("ml-ttt-data.csv") as csv_string:
    dataframe = pd.read_csv(csv_string)
calculate_adjacent_symbols(dataframe)
