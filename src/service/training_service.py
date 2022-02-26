"""
The service module contains the model training functionality of the project
"""

import pandas as pd
from sklearn import model_selection, ensemble

from src.service import data_service, generators


def train_model(model_number):
    """Loads training data from OpenML and trains a random forest classification model.

    Args:
        model_number: Integer value corresponding to a ML model

    Returns:
        model: Scikit Learn random forest model"""

    # Import dataset from csv
    predictive_features, target_feature = import_data_as_pandas(
        model_number, test=False
    )

    # Build random forest model and tune hyper-parameters
    random_forest = ensemble.RandomForestClassifier()
    param_grid = generators.get_param_grid()
    model = model_selection.GridSearchCV(random_forest, param_grid, n_jobs=1)
    model.fit(predictive_features, target_feature)

    return model


def import_data_as_pandas(model_number, test=False):
    """Imports the dataset from a csv file, from which a training or test set is
    taken. The sample is then encoded, resampled, and feature engineered as
    appropriate for the given model_number.

    Args:
        model_number [Integer]: Value corresponding to a ML model
        test [Boolean]: Set as True if the test dataset is required

    Returns:
        Integer[] : Array of predictive features
        Integer[] : List of target feature values
    """

    # Read the dataset from a csv file
    with open("ml-ttt-data.csv") as csv_string:
        input_data = pd.read_csv(csv_string, index_col=0)

    # Split the data into a training set and a test set
    training_set, test_set = model_selection.train_test_split(
        input_data, test_size=0.75, train_size=0.25, random_state=0
    )
    data_set = test_set if test else training_set

    # Apply any necessary manipulation
    if model_number == "1":  # Onhot encoded
        data_set = data_service.onehot_encode(data_set)

    elif model_number == "2":  # Ordinal encoded
        data_set = data_service.ordinal_encode(data_set)

    elif model_number == "3":  # Downsampled dataset
        if not test:
            data_set = data_service.ordinal_encode(
                data_service.downsample_dataset(data_set)
            )
        else:
            data_set = data_service.ordinal_encode(data_set)

    elif model_number == "4":  # Upsampled dataset
        if not test:
            data_set = data_service.ordinal_encode(
                data_service.upsample_dataset(data_set)
            )
        else:
            data_set = data_service.ordinal_encode(data_set)

    elif model_number == "5":  # Number of x, o, b
        data_set = data_service.ordinal_encode(
            data_service.calculate_move_counts(data_set)
        )

    elif model_number == "6":  # Adjacent pairs of x, o, b
        data_set = data_service.calculate_adjacent_symbols(data_set).iloc[:, 9:]

    elif model_number == "7":  # Best model
        if not test:
            data_set = data_service.ordinal_encode(
                data_service.calculate_adjacent_symbols(
                    data_service.upsample_dataset(data_set)
                )
            )
        else:
            data_set = data_service.ordinal_encode(
                data_service.calculate_adjacent_symbols(data_set)
            )

    else:
        raise ValueError("Invalid model_number")

    predictive_features = data_set.iloc[:, :].values
    target_feature = data_set.index.values

    return predictive_features, target_feature
