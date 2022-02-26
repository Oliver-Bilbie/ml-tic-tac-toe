"""
The API module defines the Flask app and handles the direct inputs and outputs
"""

from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
from src.service import controller


app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route("/get-prediction/<board_state>/<model_number>")
@cross_origin(supports_credentials=True)
def get_prediction(board_state, model_number):
    """
    Returns a string containing the model's prediction for the given inputs.

    Args:
        board_state: string containing the board state from top-left to bottom-right.
                     where 'x' == cross, 'o' == nought, 'b' == blank
    """

    try:
        prediction = controller.get_prediction(board_state, model_number)
        return jsonify(status=200, message=prediction)

    except ValueError:
        return jsonify(status=400, message="Invalid request")

    except Exception:
        return jsonify(status=500, message="Server was unable to process the request")


@app.route("/train-model/<model_number>")
@cross_origin(supports_credentials=True)
def train_model(model_number):
    """Trains the model from scratch.

    Args:
        model_number: Integer value corresponding to a ML model."""

    try:
        controller.train_model(model_number)
        return jsonify(status=200, message="Model successfully trained")

    except ValueError:
        return jsonify(status=400, message="Invalid request")

    except Exception:
        return jsonify(status=500, message="Server was unable to process the request")


@app.route("/test-model/<model_number>")
@cross_origin(supports_credentials=True)
def test_model(model_number):
    """Returns performance metrics for a given model

    Args:
        model_number: Integer value corresponding to a ML model."""

    try:
        results = controller.test_model(model_number)
        return jsonify(status=200, message=results)

    except ValueError:
        return jsonify(status=400, message="Invalid request")

    except Exception:
        return jsonify(status=500, message="Server was unable to process the request")
