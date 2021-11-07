"""
The service module contains the core functionality of the project
"""

import pickle
import pandas as pd
import numpy as np
from sklearn import model_selection, ensemble


class DataSet:
    """Object for handling test and training values of a dataset.

    Args:
        X: list of inputs
        y: list of outputs corresponding to the given inputs
    """

    def __init__(self, x, y):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x, y, random_state=0
        )
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


def import_data_as_pandas():
    """Import a csv file and returns it as a DataSet object as defined above.

    Returns:
        dataset: Object containing test and training sets of data for both
                 the predictive features (x) and target feature (y)
    """

    with open("ml-ttt-data.csv") as csv_string:
        dataframe = pd.read_csv(csv_string)

    predictive_features = dataframe.iloc[:, 1:]
    target = dataframe.iloc[:, 0]
    dataset = DataSet(predictive_features, target)

    return dataset


def train_model(model_number):
    """Loads training data from OpenML and trains a random forest classification model.

    Args:
        model_number: Integer value corresponding to a ML model

    Returns:
        model: Scikit Learn random forest model"""

    # Import dataset from OpenML
    game_states = import_data_as_pandas()

    # One-Hot encode the predictive features
    game_states.x_train = pd.get_dummies(game_states.x_train)

    # Build random forest model and tune hyper-parameters
    random_forest = ensemble.RandomForestClassifier()
    param_grid = get_param_grid()
    model = model_selection.GridSearchCV(random_forest, param_grid, n_jobs=1)
    model.fit(game_states.x_train, game_states.y_train)

    return model


def get_param_grid():
    """Returns a range of parameters used to optimize the model.
    Factored out for ease of testing.

    Returns:
        param_grid: Dictionary containing parameter range"""

    param_grid = {
        "n_estimators": [10, 50, 100, 250],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [4, 8, 16, 32, 64, 128, 256],
        "criterion": ["gini", "entropy"],
    }

    return param_grid


def handle_user_input(board_state):
    """Converts raw user inputs into a Pandas Dataframe object
       which may be used with the predictive model.

    Args:
        board_state: string containing the board state from top-left to bottom-right.
                     where 'x' == cross, 'o' == nought, 'b' == blank

    Returns:
        input_df: Pandas Dataframe containing reformatted and one-hot encoded user inputs.
    """

    column_names = [
        "top-left-square_x",
        "top-middle-square_x",
        "top-right-square_x",
        "middle-left-square_x",
        "middle-middle-square_x",
        "middle-right-square_x",
        "bottom-left-square_x",
        "bottom-middle-square_x",
        "bottom-right-square_x",
        "top-left-square_o",
        "top-middle-square_o",
        "top-right-square_o",
        "middle-left-square_o",
        "middle-middle-square_o",
        "middle-right-square_o",
        "bottom-left-square_o",
        "bottom-middle-square_o",
        "bottom-right-square_o",
        "top-left-square_b",
        "top-middle-square_b",
        "top-right-square_b",
        "middle-left-square_b",
        "middle-middle-square_b",
        "middle-right-square_b",
        "bottom-left-square_b",
        "bottom-middle-square_b",
        "bottom-right-square_b",
    ]

    # Create 1x27 Dataframe with all zero values
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


def save_model_to_file(model, model_number):
    """Saves a model object to a file using pickle.

    Args:
        model: SKLearn model to be saved
        model_number: Integer value corresponding to a ML model"""

    file_name = get_file_name(model_number)

    with open(file_name, "wb") as file:
        pickle.dump(model, file)


def load_model_from_file(model_number):
    """Loads a pre-trained model object using pickle.

    Args:
        model_number: Integer value corresponding to a ML model

    Returns:
        model: SKLearn model"""

    file_name = get_file_name(model_number)

    with open(file_name, "rb") as file:
        model = pickle.load(file)

    return model


def get_file_name(model_number):
    """Generates a file name corresponding to a model.

    Args:
        model_number: Integer value corresponding to a ML model

    Returns:
        file_name: File name as a string"""

    file_name = f"models/model_{str(model_number)}.pkl"

    return file_name
