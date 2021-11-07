import pytest
from unittest import mock
import os
import pandas as pd

import src.service.service as service


def test_import_data_as_pandas():
    """check the dimensions of the loaded data are as expected"""

    response = service.import_data_as_pandas()

    assert response.x_train.shape[0] + response.x_test.shape[0] == 19683
    assert response.y_train.shape[0] + response.y_test.shape[0] == 19683
    assert response.x_train.shape[1] == 9


@mock.patch(
    "src.service.service.model_selection.GridSearchCV", return_value=mock.MagicMock()
)
@mock.patch("src.service.service.ensemble.RandomForestClassifier", return_value="rf")
@mock.patch("src.service.service.pd.get_dummies", return_value=mock.MagicMock())
@mock.patch("src.service.service.import_data_as_pandas", return_value=mock.MagicMock())
def test_train_model(
    mock_import_data, mock_get_dummies, mock_rand_forest, mock_GridSearchCV
):
    param_grid = {
        "n_estimators": [10, 50, 100, 250],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [4, 8, 16, 32, 64, 128, 256],
        "criterion": ["gini", "entropy"],
    }

    service.train_model(0)

    mock_import_data.assert_called_once()
    mock_get_dummies.assert_called_once()
    mock_rand_forest.assert_called_once()
    mock_GridSearchCV.assert_called_once_with("rf", param_grid, n_jobs=1)


def test_handle_user_input_x():
    """Test with all "x" values"""

    board_state = "xxxxxxxxx"

    response = service.handle_user_input(board_state)

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
    expected_response = pd.DataFrame(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        index=column_names,
    ).transpose()

    pd.testing.assert_frame_equal(response, expected_response)


def test_handle_user_input_o():
    """Test with all "o" values"""

    board_state = "ooooooooo"

    response = service.handle_user_input(board_state)

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
    expected_response = pd.DataFrame(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        index=column_names,
    ).transpose()

    pd.testing.assert_frame_equal(response, expected_response)


def test_handle_user_input_b():
    """Test with all "b" values"""

    board_state = "bbbbbbbbb"

    response = service.handle_user_input(board_state)

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
    expected_response = pd.DataFrame(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        index=column_names,
    ).transpose()

    pd.testing.assert_frame_equal(response, expected_response)


def test_handle_user_input_mixed():
    """Test with all mixed values"""

    board_state = "xobboxobx"

    response = service.handle_user_input(board_state)

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
    expected_response = pd.DataFrame(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        index=column_names,
    ).transpose()

    pd.testing.assert_frame_equal(response, expected_response)


def test_handle_user_input_invalid():
    """Test that invalid inputs throw an exception"""

    board_state = "xobboxobz"

    exception_message = "Invalid input character"

    with pytest.raises(Exception) as re:
        service.handle_user_input(board_state)
        assert exception_message == str(re.value)


@mock.patch("src.service.service.pickle.dump")
@mock.patch(
    "src.service.service.get_file_name", return_value="src/test/resources/temp.pkl"
)
def test_save_model_to_file(mock_get_file_name, mock_dump):
    """
    Test that pickle.dump is called and a file is produces.
    This file is then deleted.
    """

    success = True
    model = None
    model_number = 0

    service.save_model_to_file(model, model_number)

    mock_get_file_name.assert_called_once_with(model_number)
    mock_dump.assert_called_once()

    try:
        os.remove("src/test/resources/temp.pkl")
    except:
        success = False

    assert success


@mock.patch("src.service.service.pickle.load")
@mock.patch(
    "src.service.service.get_file_name", return_value="src/test/resources/model.pkl"
)
def test_load_model_from_file(mock_get_file_name, mock_load):
    """Test that pickle.load is called"""

    model_number = 0

    service.load_model_from_file(model_number)

    mock_get_file_name.assert_called_once_with(model_number)
    mock_load.assert_called_once()


def test_get_file_name_1():
    """Test the function with an input of 1"""

    model_number = 1
    expected_file_name = "models/model_1.pkl"
    file_name = service.get_file_name(model_number)

    assert file_name == expected_file_name


def test_get_file_name_12345():
    """Test the function with an input of 12345"""

    model_number = 12345
    expected_file_name = "models/model_12345.pkl"
    file_name = service.get_file_name(model_number)

    assert file_name == expected_file_name
