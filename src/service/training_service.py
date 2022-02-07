"""
The service module contains the model training functionality of the project
"""

import pandas as pd
import numpy as np
from sklearn import model_selection, ensemble

from src.service import data_service, generators


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


def train_model(model_number):
    """Loads training data from OpenML and trains a random forest classification model.

    Args:
        model_number: Integer value corresponding to a ML model

    Returns:
        model: Scikit Learn random forest model"""

    # Import dataset from OpenML
    game_states = import_data_as_pandas(model_number)

    # One-Hot encode the predictive features
    # game_states.x_train = pd.get_dummies(game_states.x_train)

    # Build random forest model and tune hyper-parameters
    random_forest = ensemble.RandomForestClassifier()
    param_grid = generators.get_param_grid()
    model = model_selection.GridSearchCV(random_forest, param_grid, n_jobs=1)
    model.fit(game_states.x_train, game_states.y_train)

    return model


def import_data_as_pandas(model_number):
    """Import a csv file and returns it as a DataSet object as defined above.

    Args:
        model_number: Integer value corresponding to a ML model

    Returns:
        dataset: Object containing test and training sets of data for both
                 the predictive features (x) and target feature (y)
    """

    with open("ml-ttt-data.csv") as csv_string:
        dataframe = pd.read_csv(csv_string, index_col=0)

    mapper = {
        # No feature engineering
        "1": data_service.onehot_encode(dataframe),
        # Balanced dataset
        "2": data_service.balance_dataset(data_service.onehot_encode(dataframe)),
        # Number of x, o, b
        "3": data_service.calculate_move_counts(dataframe),
        # Adjacent pairs of x, o, b
        "4": data_service.calculate_adjacent_symbols(dataframe),
        # Best model without "cheating"
        "5": dataframe,
    }

    dataframe = mapper.get(model_number)

    predictive_features = dataframe.iloc[:, :]
    target = dataframe.index.values
    dataset = DataSet(predictive_features, target)

    return dataset
