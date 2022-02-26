from unittest import mock

from src.service import training_service


@mock.patch(
    "src.service.training_service.model_selection.GridSearchCV",
    return_value=mock.MagicMock(),
)
@mock.patch(
    "src.service.training_service.ensemble.RandomForestClassifier", return_value="rf"
)
@mock.patch(
    "src.service.training_service.import_data_as_pandas", return_value=mock.MagicMock()
)
def test_train_model(mock_import_data, mock_rand_forest, mock_GridSearchCV):
    param_grid = {
        "n_estimators": [50, 100, 250],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [4, 8, 16, 32, 64, 128, 256],
        "criterion": ["gini", "entropy"],
    }

    training_service.train_model(0)

    mock_import_data.assert_called_once()
    mock_rand_forest.assert_called_once()
    mock_GridSearchCV.assert_called_once_with("rf", param_grid, n_jobs=1)


def test_import_data_as_pandas():
    """check the dimensions of the loaded data are as expected"""

    response = training_service.import_data_as_pandas("1")

    assert response.x_train.shape[0] + response.x_test.shape[0] == 19683
    assert response.y_train.shape[0] + response.y_test.shape[0] == 19683
    assert response.x_train.shape[1] == 10
