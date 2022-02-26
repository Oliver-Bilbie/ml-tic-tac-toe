"""
The validator module is used to verify that user inputs are valid
"""


def validate_board_state(board_state):
    """Raises an exception if the input is not a nine-character string
    containing only characters from {"x", "o", "b"}

    Args:
        board_state: string corresponding to a board state."""

    try:
        if len(board_state) != 9:
            raise ValueError("Validation Error: Incorrect length")

        for char in board_state:
            if not char in ["x", "o", "b"]:
                raise ValueError("Validation Error: Invalid character")

    except:
        raise ValueError("Validation Error: Invalid input") from Exception


def validate_model_number(model_number):
    """Raises an exception if the input is not a valid model number.

    Args:
        model_number: string corresponding to a model number."""

    if not model_number in ["1", "2", "3", "4", "5", "6", "7"]:
        raise ValueError("Validation Error: Invalid model reference")
