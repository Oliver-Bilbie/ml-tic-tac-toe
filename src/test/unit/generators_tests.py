from src.service import generators


def test_get_onehot_column_names():
    expected_cols = [
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

    actual_cols = generators.get_onehot_column_names()

    assert expected_cols == actual_cols


def test_get_param_grid():
    expected_grid = {
        "n_estimators": [50, 100, 250],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [4, 8, 16, 32, 64, 128, 256],
        "criterion": ["gini", "entropy"],
    }

    actual_grid = generators.get_param_grid()

    assert expected_grid == actual_grid
