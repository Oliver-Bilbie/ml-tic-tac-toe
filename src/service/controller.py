"""
The controller module acts as an interface between the user inputs and the service layers
"""

from src.service import (
    prediction_service,
    training_service,
    testing_service,
    file_service,
    validator,
)


def get_prediction(board_state, model_number):
    """Predict the outcome of a game given its board state

    Args:
        board_state: string containing the board state from top-left to bottom-right.
                     where 'x' == cross, 'o' == nought, 'b' == blank
        model_number: Integer value corresponding to a ML model.

    Returns:
         prediction: String containing the model's prediction for the given inputs."""

    validator.validate_board_state(board_state)
    validator.validate_model_number(model_number)

    user_input = prediction_service.handle_user_input(board_state, model_number)
    model = file_service.load_model_from_file(model_number)
    prediction = prediction_service.evaluate_prediction(model, user_input)

    return prediction


def train_model(model_number):
    """Train a ML model and save it to a file

    Args:
        model_number: Integer value corresponding to a ML model"""

    validator.validate_model_number(model_number)
    model = training_service.train_model(model_number)
    file_service.save_model_to_file(model, model_number)


def test_model(model_number):
    """Evaluate the f1 score, precision, recall, and confusion matrix of a model

    Args:
        model_number: Integer value corresponding to a ML model"""

    validator.validate_model_number(model_number)
    model = file_service.load_model_from_file(model_number)
    predictive_features, target_feature = training_service.import_data_as_pandas(
        model_number, test=True
    )
    metrics = testing_service.test_model(model, predictive_features, target_feature)

    return metrics
