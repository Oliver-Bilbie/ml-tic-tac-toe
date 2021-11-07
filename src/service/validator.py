"""
The validator module is used to verify that user inputs are valid
"""


def validate_prediction_request(request):
    """Raises an exception if the input is not a nine-character string
    containing only characters from {"x", "o", "b"}

    Args:
        request: string corresponding to a board state."""

    try:
        if len(request) != 9:
            raise ValueError("Validation Error: Incorrect length")

        for char in request:
            if not char in ["x", "o", "b"]:
                raise ValueError("Validation Error: Invalid character")

    except:
        raise ValueError("Validation Error: Invalid input") from Exception


def validate_train_request(request):
    """Raises an exception if the input is not a valid model number.

    Args:
        request: string corresponding to a mmodel number."""

    if not request in ["1"]:
        raise ValueError("Validation Error: Invalid model reference")
