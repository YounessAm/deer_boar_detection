import joblib
from flask import Flask, request, json, jsonify, render_template
from werkzeug.exceptions import HTTPException
from cerf_sanglier_detection.yolo_inference import detect_animal, YOLO_WEIGHTS_PATH


app = Flask(__name__)


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors (which is the basic error
    response with Flask).
    """
    # Start with the correct headers and status code from the error
    response = e.get_response()
    # Replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


class MissingKeyError(HTTPException):
    # We can define our own error for the missing key
    code = 422
    name = "Missing key error"
    description = "JSON content missing key 'input'."


class MissingJSON(HTTPException):
    # We can define our own error for missing JSON
    code = 400
    name = "Missing JSON"
    description = "Missing JSON."


class BadInputType(HTTPException):
    # We can define our own error for missing JSON
    code = 425
    name = "Input_type error"
    description = "the input must be a list or a string"



class BadType(HTTPException):
    # We can define our own error for missing JSON
    code = 435
    name = "List input_Type error"
    description = "the input must be a string or a list of strings "


def good_format(input):
    try:
        return sum([isinstance(i,str) for i in input])==len(input)
    except :
        return False

def is_str(x):
    return isinstance(x,str)


@app.route("/predict", methods=["POST"])
def predict():
    # Check parameters
    if request.json:
        # Get JSON as dictionnary
        json_input = request.get_json()
        if "input" not in json_input:
            # If 'input' is not in our JSON we raise our own error
            raise MissingKeyError()
        
        input=json_input["input"]
        # check the input and call our predict function that handle loading model and making a        
        if good_format(input):
            # prediction
            #print(json_input["input"])
            print('input')
            prediction = detect_animal(input, YOLO_WEIGHTS_PATH, confidence=0.25)
            # Return prediction
            # response = json.dump(prediction)
            return prediction, 200

        else : 
            raise BadInputType()

    raise MissingJSON()


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
