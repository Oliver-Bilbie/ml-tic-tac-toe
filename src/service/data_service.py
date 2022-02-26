"""
The data service module contains the logic for feature engineering and data manipulation
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample

from src.service import generators


def onehot_encode(dataset):
    """Onehot encodes the board state features while preserving additional features

    Args:
        dataset [DataFrame]: Pandas DataFrame containing the dataset to be onehot encoded

    Returns:
        [DataFrame]: Onehot encoded dataset
    """

    columns_to_encode = generators.get_board_state_column_names()

    # Preserve additional columns
    additional_columns = []
    for name in dataset.columns:
        if not name in columns_to_encode:
            additional_columns.append(name)

    # Append additional entries to ensure that all posibilities are one-hot encoded
    temp_rows = pd.DataFrame(
        {
            "top-left-square": ["x", "o", "b"],
            "top-middle-square": ["x", "o", "b"],
            "top-right-square": ["x", "o", "b"],
            "middle-left-square": ["x", "o", "b"],
            "middle-middle-square": ["x", "o", "b"],
            "middle-right-square": ["x", "o", "b"],
            "bottom-left-square": ["x", "o", "b"],
            "bottom-middle-square": ["x", "o", "b"],
            "bottom-right-square": ["x", "o", "b"],
        },
        index=["temp", "temp", "temp"],
    )
    dataset = pd.concat([dataset, temp_rows])

    # Encode dataset
    onehot_dataset = pd.get_dummies(data=dataset[columns_to_encode])

    # Re-assemble the dataset and remove the temporary data items
    additional_dataset = dataset[additional_columns]
    complete_dataset = pd.concat([onehot_dataset, additional_dataset], axis=1)
    complete_dataset = complete_dataset[complete_dataset.index != "temp"]

    return complete_dataset


def ordinal_encode(dataset):
    """Numerically encodes the board state features such that:
        x => 1
        b => 0
        o => -1

    Args:
        dataset [DataFrame]: Pandas DataFrame containing the dataset to be encoded

    Returns:
        [DataFrame]: Numerically encoded dataset
    """

    dataset = dataset.replace(["x", "b", "o"], [1, 0, -1])

    return dataset


def downsample_dataset(dataset):
    """Reduces the number of data items such that all outcomes are equally represented

    Args:
        dataset [DataFrame]: Pandas DataFrame containing the dataset to be downsampled

    Returns:
        [DataFrame]: Downsampled dataset
    """

    value_counts = dataset.index.value_counts(sort=True)
    target_size = value_counts[-1]
    outcomes_to_downsample = value_counts.index[:-1]

    # List of datasets to be combined once everything has been downsampled
    # We initially add the smallest item since this is not going to be downsampled
    downsampled_datasets = [dataset[dataset.index == value_counts.index[-1]]]

    for outcome in outcomes_to_downsample:
        downsampled_datasets.append(
            resample(
                dataset[dataset.index == outcome],
                replace=False,
                n_samples=target_size,
                random_state=0,
            )
        )

    dataset = pd.concat(downsampled_datasets)

    return dataset


def upsample_dataset(dataset):
    """Increases the number of data items such that all outcomes are equally represented.
    This is achieved by duplication of under-represented outcomes.

    Args:
        dataset [DataFrame]: Pandas DataFrame containing the dataset to be upsampled

    Returns:
        [DataFrame]: Upsampled dataset
    """

    value_counts = dataset.index.value_counts(sort=True)
    target_size = value_counts[0]
    outcomes_to_upsample = value_counts.index[1:]

    # List of datasets to be combined once everything has been upsampled
    # We initially add the largest item since this is not going to be upsampled
    upsampled_datasets = [dataset[dataset.index == value_counts.index[0]]]

    for outcome in outcomes_to_upsample:
        upsampled_datasets.append(
            resample(
                dataset[dataset.index == outcome],
                replace=True,
                n_samples=target_size,
                random_state=0,
            )
        )

    dataset = pd.concat(upsampled_datasets)

    return dataset


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
    x_df = dataset.copy().replace(["x", "o", "b"], [1, 0, 0])
    # Sum each row
    x_count = x_df.sum(axis=1)

    # Replace all O values with 1 and all other values with 0
    o_df = dataset.copy().replace(["x", "o", "b"], [0, 1, 0])
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
