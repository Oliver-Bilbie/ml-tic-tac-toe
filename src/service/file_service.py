"""
The file service module contains the logic for handling the saving/loading of models to/from files
"""

import pickle


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
