"""
The service module contains the prediction functionality of the project
"""

import pandas as pd
import numpy as np

from src.service import data_service, generators


def handle_user_input(board_state, model_number):
    """Converts raw user inputs into a Pandas Dataframe object
    which may be used with the predictive model.

    Args:
        board_state [String]: string containing the board state from top-left to bottom-right.
                              where 'x' == cross, 'o' == nought, 'b' == blank
        model_number [String]: Integer value corresponding to a ML model.

    Returns:
        [DataFrame]: Pandas Dataframe containing reformatted and one-hot encoded user inputs.
    """

    column_names = generators.get_board_state_column_names()
    input_df = pd.DataFrame(np.zeros(9), index=column_names).transpose()
    for square_index in range(0, 9):
        input_df.iloc[0, square_index] = board_state[square_index]

    # Apply any necessary manipulation
    if model_number == "1":
        input_df = data_service.onehot_encode(input_df)
    elif model_number == "5":
        input_df = data_service.ordinal_encode(
            data_service.calculate_move_counts(input_df)
        )
    elif model_number == "6":
        input_df = data_service.calculate_adjacent_symbols(input_df).iloc[:, 9:]
    elif model_number == "7":
        input_df = data_service.ordinal_encode(
            data_service.calculate_adjacent_symbols(input_df)
        )
    else:
        input_df = data_service.ordinal_encode(input_df)

    input_df = data_service.ordinal_encode(input_df)

    return input_df


def evaluate_prediction(model, user_input):

    prediction = model.predict(user_input)

    # remove square brackets and apostrophes from the prediction
    prediction_string = str(prediction)[2:-2]

    return prediction_string
