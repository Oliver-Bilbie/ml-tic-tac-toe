from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
import src.controller as controller
import src.validator as validator

app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route('/get-prediction/<board_state>')
@cross_origin(supports_credentials=True)
def get_prediction(board_state):
    """
    Returns a string containing the model's prediction for the given inputs.
    
    Args:
        board_state: string containing the board state from top-left to bottom-right.
                     where 'x' == cross, 'o' == nought, 'b' == blank
    """

    try:
        validator.validate_request(board_state)

        tl = board_state[0]
        tm = board_state[1]
        tr = board_state[2]
        ml = board_state[3]
        mm = board_state[4]
        mr = board_state[5]
        bl = board_state[6]
        bm = board_state[7]
        br = board_state[8]

        prediction = controller.get_prediction(tl, tm, tr, ml, mm, mr, bl, bm, br)

        return jsonify(status=200, message=prediction)
    
    except:
        return jsonify(status=500, message="Server was unable to process the request")


@app.route('/train-model')
@cross_origin(supports_credentials=True)
def train_model():
    """Trains the model from scratch."""

    try:
        controller.train_model()
        return jsonify(status=200, message="Model successfully trained")
    
    except:
        return jsonify(status=500, message="Server was unable to process the request")
