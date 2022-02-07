import pytest
from unittest import mock
import os

from src.service import api, controller


@mock.patch("src.service.api.jsonify")
@mock.patch(
    "src.service.api.controller.training_service.generators.get_param_grid",
    return_value={
        "n_estimators": [10],
        "max_features": ["sqrt"],
        "max_depth": [4],
        "criterion": ["entropy"],
    },
)
@mock.patch(
    "src.service.api.controller.file_service.get_file_name",
    return_value="src/test/resources/temp.pkl",
)
def test_train_model(mock_get_file_name, mock_param_grid, mock_jsonify):
    """Test that the train_model api produces a file. This file is then deleted."""

    train_model = api.train_model.__wrapped__
    success = True
    model_number = "1"

    try:
        train_model(model_number)
        os.remove("src/test/resources/temp.pkl")
    except:
        success = False

    assert success


@mock.patch(
    "src.service.controller.file_service.get_file_name",
    return_value="src/test/resources/model.pkl",
)
def test_controller_get_prediction(mock_get_file_name):
    """Test that get_prediction in the controller returns a valid response (not necessarily correct!)"""

    board_state = "bbbbbbbbb"
    model_number = "1"

    response = controller.get_prediction(board_state, model_number)

    valid_responses = ["nobody", "x", "o", "everyone"]

    assert response in valid_responses


@mock.patch("src.service.api.jsonify")
@mock.patch(
    "src.service.api.controller.file_service.get_file_name",
    return_value="src/test/resources/model.pkl",
)
def test_get_prediction(mock_get_file_name, mock_jsonify):
    """Test that the get_prediction api returns an appropriate response for a successful request"""

    get_prediction = api.get_prediction.__wrapped__
    board_state = "bbbbbbbbb"
    model_number = "1"

    get_prediction(board_state, model_number)

    mock_jsonify.assert_called_once_with(status=200, message=mock.ANY)


@mock.patch("src.service.api.jsonify")
@mock.patch(
    "src.service.api.controller.file_service.get_file_name",
    return_value="src/test/resources/model.pkl",
)
def test_get_prediction_invalid(mock_get_file_name, mock_jsonify):
    """Test that the get_prediction api returns an appropriate response for an invalid request"""

    get_prediction = api.get_prediction.__wrapped__
    board_state = "abcdefghi"
    model_number = "1"

    get_prediction(board_state, model_number)

    mock_jsonify.assert_called_once_with(status=400, message="Invalid request")


@mock.patch("src.service.api.jsonify")
@mock.patch(
    "src.service.api.controller.file_service.get_file_name",
    side_effect=Exception,
)
def test_get_prediction_unsuccessful(mock_get_file_name, mock_jsonify):
    """Test that the get_prediction api returns an appropriate response for an unsuccessful request"""

    get_prediction = api.get_prediction.__wrapped__
    board_state = "bbbbbbbbb"
    model_number = "1"

    get_prediction(board_state, model_number)

    mock_jsonify.assert_called_once_with(
        status=500, message="Server was unable to process the request"
    )
