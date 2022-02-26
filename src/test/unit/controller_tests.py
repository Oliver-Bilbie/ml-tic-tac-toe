from unittest import mock

from src.service import controller


class mock_model:
    def predict(self, user_input):
        return "['response']"


@mock.patch(
    "src.service.controller.file_service.load_model_from_file",
    return_value=mock_model(),
)
@mock.patch(
    "src.service.controller.prediction_service.handle_user_input",
    return_value="user_input",
)
@mock.patch("src.service.controller.validator.validate_board_state")
def test_get_prediction(
    mock_validate_board_state, mock_handle_user_input, mock_load_model_from_file
):
    """
    Test for a successful request
    """

    board_state = "123456789"
    model_number = "1"
    response = controller.get_prediction(board_state, model_number)

    mock_validate_board_state.assert_called_once_with(board_state)
    mock_handle_user_input.assert_called_once_with(board_state)
    mock_load_model_from_file.assert_called_once_with(model_number)

    assert response == "response"


@mock.patch("src.service.controller.file_service.save_model_to_file")
@mock.patch("src.service.controller.training_service.train_model", return_value="model")
@mock.patch("src.service.controller.validator.validate_model_number")
def test_train_model(
    mock_validate_model_number, mock_train_model, mock_save_model_to_file
):
    """
    Test for a successful request
    """

    controller.train_model(0)
    mock_validate_model_number.assert_called_once_with(0)
    mock_train_model.assert_called_once_with(0)
    mock_save_model_to_file.assert_called_once_with("model", 0)
