import src.service as service


def get_prediction(tl, tm, tr, ml, mm, mr, bl, bm, br):
    """Predict the outcome of a game given its board state

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
         prediction: String containing the model's prediction for the given inputs."""

    model = service.load_model_from_file()

    user_input = service.handle_user_input(tl, tm, tr, ml, mm, mr, bl, bm, br)

    prediction = model.predict(user_input)

    return str(prediction)


def train_model():
    """Train the ML model"""

    model = service.train_model()
    service.save_model_to_file(model)
