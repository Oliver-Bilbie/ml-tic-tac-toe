import pytest
import pandas as pd

from src.service import data_service


@pytest.fixture
def mock_dataset():
    def _inner():
        with open("ml-ttt-data.csv") as csv_string:
            dataframe = pd.read_csv(csv_string, index_col=0)

        return dataframe

    return _inner()


def test_onehot_encode(mock_dataset):
    response = data_service.onehot_encode(mock_dataset)

    assert response.shape[0] == 19683
    assert response.shape[1] == 27


def test_write_tests():
    assert False
