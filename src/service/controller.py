"""
The controller module acts as an interface between the user inputs and the service layer
"""

from src.service import service, validator


def get_prediction(board_state, model_number):
    """Predict the outcome of a game given its board state

    Args:
        board_state: string containing the board state from top-left to bottom-right.
                     where 'x' == cross, 'o' == nought, 'b' == blank

    Returns:
         prediction: String containing the model's prediction for the given inputs."""

    validator.validate_prediction_request(board_state)

    user_input = service.handle_user_input(board_state)
    model = service.load_model_from_file(model_number)
    prediction = model.predict(user_input)
    # remove square brackets and apostrophes from the prediction
    prediction_string = str(prediction)[2:-2]

    return prediction_string


def train_model(model_number):
    """Train a ML model

    Args:
        model_number: Integer value corresponding to a ML model"""

    validator.validate_train_request(model_number)
    model = service.train_model(model_number)
    service.save_model_to_file(model, model_number)
