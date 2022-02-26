from unittest import mock
import os

from src.service import file_service


@mock.patch("src.service.file_service.pickle.dump")
@mock.patch(
    "src.service.file_service.get_file_name", return_value="src/test/resources/temp.pkl"
)
def test_save_model_to_file(mock_get_file_name, mock_dump):
    """
    Test that pickle.dump is called and a file is produces.
    This file is then deleted.
    """

    success = True
    model = None
    model_number = 0

    file_service.save_model_to_file(model, model_number)

    mock_get_file_name.assert_called_once_with(model_number)
    mock_dump.assert_called_once()

    try:
        os.remove("src/test/resources/temp.pkl")
    except:
        success = False

    assert success


@mock.patch("src.service.file_service.pickle.load")
@mock.patch(
    "src.service.file_service.get_file_name",
    return_value="src/test/resources/model.pkl",
)
def test_load_model_from_file(mock_get_file_name, mock_load):
    """Test that pickle.load is called"""

    model_number = 0

    file_service.load_model_from_file(model_number)

    mock_get_file_name.assert_called_once_with(model_number)
    mock_load.assert_called_once()


def test_get_file_name_1():
    """Test the function with an input of 1"""

    model_number = 1
    expected_file_name = "models/model_1.pkl"
    file_name = file_service.get_file_name(model_number)

    assert file_name == expected_file_name


def test_get_file_name_12345():
    """Test the function with an input of 12345"""

    model_number = 12345
    expected_file_name = "models/model_12345.pkl"
    file_name = file_service.get_file_name(model_number)

    assert file_name == expected_file_name
