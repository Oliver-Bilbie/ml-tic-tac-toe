"""
The service module contains the prediction functionality of the project
"""

import pandas as pd
import numpy as np

from src.service import generators


def handle_user_input(board_state):
    """Converts raw user inputs into a Pandas Dataframe object
    which may be used with the predictive model.

    Args:
        board_state [String]: string containing the board state from top-left to bottom-right.
                              where 'x' == cross, 'o' == nought, 'b' == blank

    Returns:
        [DataFrame]: Pandas Dataframe containing reformatted and one-hot encoded user inputs.
    """

    # Create 1x27 Dataframe with all zero values
    column_names = generators.get_onehot_column_names()
    input_df = pd.DataFrame(np.zeros(27), index=column_names).transpose()

    # Populate the Dataframe with user inputs
    for input_number in range(0, 9):
        if board_state[input_number] == "x":
            column_number = input_number
        elif board_state[input_number] == "o":
            column_number = input_number + 9
        elif board_state[input_number] == "b":
            column_number = input_number + 18
        else:
            raise Exception("Invalid input character")

        input_df.iloc[0, column_number] = 1

    return input_df


def evaluate_prediction(model, user_input):

    prediction = model.predict(user_input)

    # remove square brackets and apostrophes from the prediction
    prediction_string = str(prediction)[2:-2]

    return prediction_string
