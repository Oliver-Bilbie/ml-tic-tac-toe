import pytest

import src.service.validator as validator


def test_validate_prediction_request_valid():
    """
    Test with valid requests
    """

    input_1 = "xxxxxxxxx"
    input_2 = "ooooooooo"
    input_3 = "bbbbbbbbb"
    input_4 = "xobxobxob"
    input_5 = "xbboxbobx"

    try:
        validator.validate_prediction_request(input_1)
        validator.validate_prediction_request(input_2)
        validator.validate_prediction_request(input_3)
        validator.validate_prediction_request(input_4)
        validator.validate_prediction_request(input_5)
        success = True
    except:
        success = False

    assert success


def test_validate_prediction_request_too_short():
    """
    Test with eight-character request (nine required)
    """

    input = "xxxxxxxx"

    exception_message = "Validation Error: Incorrect length"

    with pytest.raises(Exception) as re:
        validator.validate_prediction_request(input)
        assert exception_message == str(re.value)


def test_validate_prediction_request_too_long():
    """
    Test with ten-character request (nine required)
    """

    input = "xxxxxxxxxx"

    exception_message = "Validation Error: Incorrect length"

    with pytest.raises(Exception) as re:
        validator.validate_prediction_request(input)
        assert exception_message == str(re.value)


def test_validate_prediction_request_invalid_character():
    """
    Test with an invalid input
    """

    input = "xxxxxxxxy"

    exception_message = "Validation Error: Invalid character"

    with pytest.raises(Exception) as re:
        validator.validate_prediction_request(input)
        assert exception_message == str(re.value)


def test_validate_prediction_request_empty():
    """
    Test with an empty string
    """

    input = ""

    exception_message = "Validation Error: Incorrect length"

    with pytest.raises(Exception) as re:
        validator.validate_prediction_request(input)
        assert exception_message == str(re.value)


def test_validate_prediction_request_bad_type():
    """
    Test with an integer (requires a string)
    """

    input = 123456789

    exception_message = "Validation Error: Invalid input"

    with pytest.raises(Exception) as re:
        validator.validate_prediction_request(input)
        assert exception_message == str(re.value)


def test_validate_train_request_valid():
    """
    Test with a valid input value
    """

    model_number = "1"

    try:
        validator.validate_train_request(model_number)
        success = True
    except:
        success = False

    assert success


def test_validate_train_request_invalid():
    """
    Test with an invalid input value
    """

    input = "123456789"

    exception_message = "Validation Error: Invalid model reference"

    with pytest.raises(Exception) as re:
        validator.validate_train_request(input)
        assert exception_message == str(re.value)
