from unittest import mock

import src.service.api as api


@mock.patch("src.service.api.jsonify")
@mock.patch("src.service.api.controller.get_prediction", return_value="prediction")
def test_get_prediction_success(mock_controller, mock_jsonify):
    """
    Test for a successful request
    """

    get_prediction = api.get_prediction.__wrapped__
    board_state = "bbbbbbbbb"
    model_number = 1

    get_prediction(board_state, model_number)

    mock_controller.assert_called_once_with(board_state, model_number)
    mock_jsonify.assert_called_once_with(status=200, message="prediction")


@mock.patch("src.service.api.jsonify")
@mock.patch("src.service.api.controller.get_prediction", side_effect=ValueError)
def test_get_prediction_validation_error(mock_controller, mock_jsonify):
    """
    Test for an unsuccessful request due to an invalid input
    """

    get_prediction = api.get_prediction.__wrapped__
    board_state = "bbbbbbbbb"
    model_number = 1

    get_prediction(board_state, model_number)

    mock_controller.assert_called_once_with(board_state, model_number)
    mock_jsonify.assert_called_once_with(status=400, message="Invalid request")


@mock.patch("src.service.api.jsonify")
@mock.patch("src.service.api.controller.get_prediction", side_effect=Exception)
def test_get_prediction_server_error(mock_controller, mock_jsonify):
    """
    Test for an unsuccessful request due to a runtime error
    """

    get_prediction = api.get_prediction.__wrapped__
    board_state = "bbbbbbbbb"
    model_number = 1

    get_prediction(board_state, model_number)

    mock_controller.assert_called_once_with(board_state, model_number)
    mock_jsonify.assert_called_once_with(
        status=500, message="Server was unable to process the request"
    )


@mock.patch("src.service.api.jsonify")
@mock.patch("src.service.api.controller.train_model", return_value="prediction")
def test_train_model_success(mock_controller, mock_jsonify):
    """
    Test for a successful request
    """

    train_model = api.train_model.__wrapped__
    model_number = "1"

    train_model(model_number)

    mock_controller.assert_called_once_with(model_number)
    mock_jsonify.assert_called_once_with(
        status=200, message="Model successfully trained"
    )


@mock.patch("src.service.api.jsonify")
@mock.patch("src.service.api.controller.train_model", side_effect=ValueError)
def test_train_model_validation_error(mock_controller, mock_jsonify):
    """
    Test for an unsuccessful request due to an invalid input
    """

    train_model = api.train_model.__wrapped__
    model_number = "1"

    train_model(model_number)

    mock_controller.assert_called_once_with(model_number)
    mock_jsonify.assert_called_once_with(status=400, message="Invalid request")


@mock.patch("src.service.api.jsonify")
@mock.patch("src.service.api.controller.train_model", side_effect=Exception)
def test_train_model_server_error(mock_controller, mock_jsonify):
    """
    Test for an unsuccessful request due to a runtime error
    """

    train_model = api.train_model.__wrapped__
    model_number = "1"

    train_model(model_number)

    mock_controller.assert_called_once_with(model_number)
    mock_jsonify.assert_called_once_with(
        status=500, message="Server was unable to process the request"
    )
