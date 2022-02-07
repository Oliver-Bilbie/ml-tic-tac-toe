import pytest
import pandas as pd

from src.service import prediction_service


def test_handle_user_input_x():
    """Test with all "x" values"""

    board_state = "xxxxxxxxx"

    response = prediction_service.handle_user_input(board_state)

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

    response = prediction_service.handle_user_input(board_state)

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

    response = prediction_service.handle_user_input(board_state)

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

    response = prediction_service.handle_user_input(board_state)

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
        prediction_service.handle_user_input(board_state)
        assert exception_message == str(re.value)
