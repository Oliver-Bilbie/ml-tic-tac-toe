import pandas as pd
import numpy as np
import pickle
from sklearn import model_selection, ensemble


class DataSet:
    """Object for handling test and training values of a dataset.

    Args:
        X: list of inputs
        y: list of outputs corresponding to the given inputs
    """

    def __init__(self, X, y):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


def import_dataset_as_pandas():
    """Import a csv file and returns it as a DataSet object as defined above.

    Returns:
        dataset: Object containing test and training sets of data for both
                 the predictive features (X) and target feature (y)
    """

    csv_string = open("ml-ttt-data.csv")
    dataframe = pd.read_csv(csv_string)
    X = dataframe.iloc[:, 1:]
    y = dataframe.iloc[:, 0]
    dataset = DataSet(X, y)

    return dataset


def train_model():
    """Loads training data from OpenML and trains a random forest classification model.

    Returns:
        CV: Scikit Learn random forest model"""

    # Import dataset from OpenML
    game_states = import_dataset_as_pandas()

    # One-Hot encode the predictive features
    game_states.X_train = pd.get_dummies(game_states.X_train)

    # Build random forest model and tune hyper-parameters
    rf = ensemble.RandomForestClassifier()
    param_grid = {
        'n_estimators': [10, 50, 100, 250],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 8, 16, 32, 64, 128, 256],
        'criterion': ['gini', 'entropy']
    }
    CV = model_selection.GridSearchCV(rf, param_grid, n_jobs=1)
    CV.fit(game_states.X_train, game_states.y_train)

    return CV


def handle_user_input(tl, tm, tr, ml, mm, mr, bl, bm, br):
    """Converts raw user inputs into a Pandas Dataframe object which may be used with the predictive model.

    Args:
        tl: String from {"x", "o", "b"} corresponding to the top-left tile's value.
        tm: String from {"x", "o", "b"} corresponding to the top-middle tile's value.
        tr: String from {"x", "o", "b"} corresponding to the top-right tile's value.
        ml: String from {"x", "o", "b"} corresponding to the middle-left tile's value.
        mm: String from {"x", "o", "b"} corresponding to the middle-middle tile's value.
        mr: String from {"x", "o", "b"} corresponding to the middle-right tile's value.
        bl: String from {"x", "o", "b"} corresponding to the bottom-left tile's value.
        bm: String from {"x", "o", "b"} corresponding to the bottom-middle tile's value.
        br: String from {"x", "o", "b"} corresponding to the bottom-right tile's value.

    Returns:
        input_df: Pandas Dataframe containing reformatted and one-hot encoded user inputs.
    """
    user_inputs = [
        tl, tm, tr,
        ml, mm, mr,
        bl, bm, br
    ]

    column_names = [
        'top-left-square_x',
        'top-middle-square_x',
        'top-right-square_x',
        'middle-left-square_x',
        'middle-middle-square_x',
        'middle-right-square_x',
        'bottom-left-square_x',
        'bottom-middle-square_x',
        'bottom-right-square_x',
        'top-left-square_o',
        'top-middle-square_o',
        'top-right-square_o',
        'middle-left-square_o',
        'middle-middle-square_o',
        'middle-right-square_o',
        'bottom-left-square_o',
        'bottom-middle-square_o',
        'bottom-right-square_o',
        'top-left-square_b',
        'top-middle-square_b',
        'top-right-square_b',
        'middle-left-square_b',
        'middle-middle-square_b',
        'middle-right-square_b',
        'bottom-left-square_b',
        'bottom-middle-square_b',
        'bottom-right-square_b',
    ]

    # Create 1x27 Dataframe with all zero values
    input_df = pd.DataFrame(np.zeros(27), index=column_names).transpose()

    # Populate the Dataframe with user inputs
    for input_number in range(0, 9):
        if user_inputs[input_number] == "x":
            column_number = input_number
        elif user_inputs[input_number] == "o":
            column_number = input_number + 9
        elif user_inputs[input_number] == "b":
            column_number = input_number + 18
        else:
            raise Exception("Invalid input character")

        input_df.iloc[0, column_number] = 1

    return input_df


def save_model_to_file(model):
    """Saves a model object to a file using pickle.
    
    Args:
        model: SKLearn model to be saved"""

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)


def load_model_from_file():
    """Loads a pre-trained model object using pickle.
    
    Returns:
        model: SKLearn model"""

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    return model
