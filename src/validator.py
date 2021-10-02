def validate_request(body):
    if len(body) != 9:
        raise Exception("Validation Error")

    for char in body:
        if char != "x" and char != "o" and char != "b":
            raise Exception("Validation Error")
