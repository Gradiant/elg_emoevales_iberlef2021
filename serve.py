from flask import Flask, request
from flask_json import FlaskJSON, JsonError, as_json
from werkzeug.utils import secure_filename
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import json
from scipy.special import softmax

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
APP_ROOT = "./"
app.config["APPLICATION_ROOT"] = APP_ROOT
app.config["UPLOAD_FOLDER"] = "files/"

json_app = FlaskJSON(app)

model_path = "daveni/twitter-xlm-roberta-emotion-es"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def tokenize(text):
    text = preprocess(text)
    encoded_input = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    )
    return encoded_input


def predict(encoded_input):
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores


@as_json
@app.route("/predict_json", methods=["POST"])
def predict_json():

    data = request.get_json()
    if (data.get("type") != "text") or ("content" not in data):
        output = invalid_request_error(None)
        return output
    content = data["content"]
    try:
        encoded_input = tokenize(content)
        size = len(encoded_input["input_ids"][0])
        scores = predict(encoded_input)
        output = generate_successful_response(scores)
        return output
    except Exception as e:
        return generate_failure_response(
            status=404,
            code="elg.service.internalError",
            text=None,
            params=None,
            detail=str(e),
        )


def generate_successful_response(scores):
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    list_clasess = list()
    for i in range(scores.shape[0]):
        list_clasess.append(
            {"class": config.id2label[ranking[i]], "score": str(scores[ranking[i]])}
        )
    response = {"type": "classification", "classes": list_clasess}
    output = {"response": response}
    return output


@json_app.invalid_json_error
def invalid_request_error(e):
    """Generates a valid ELG "failure" response if the request cannot be parsed"""
    raise JsonError(
        status_=400,
        failure={
            "errors": [
                {"code": "elg.request.invalid", "text": "Invalid request message"}
            ]
        },
    )


def generate_failure_response(status, code, text, params, detail):
    error = {}
    if code:
        error["code"] = code
    if text:
        error["text"] = text
    if params:
        error["params"] = params
    if detail:
        error["detail"] = {"message": detail}

    raise JsonError(status_=status, failure={"errors": [error]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8866)
