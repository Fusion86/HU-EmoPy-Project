"""
Hosts a HTTP server which accepts (preprocessed)images and returns what emotions it found in said images.

API usage:
POST :5000/predict
Form-data:
"content"   = image_bytes (supports PNG and many others)
"grayscale" = optional, when set the content is assumed to be a grayscale image (1 channel)
"""

import io
import cv2
import numpy as np
from flask import Flask, request, jsonify
from fermodel import FERModel

# Settings
onnx_model = "model.onnx"
port = 5000

app = Flask(__name__)
model = FERModel(onnx_model)


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


def buf_to_img(req_file, color_image_flag=cv2.IMREAD_COLOR):
    in_memory_file = io.BytesIO()
    req_file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, color_image_flag)


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/")
def hello():
    return "R2D2 FER Server"


@app.route("/predict", methods=["POST"])
def predict():
    if "content" not in request.files:
        raise InvalidUsage("Missing form-data 'content'.")

    req = request

    if "grayscale" in request.form:
        image = buf_to_img(request.files["content"], cv2.IMREAD_GRAYSCALE)
    else:
        image = buf_to_img(request.files["content"])
    res = model.predict(image)

    return res


if __name__ == "__main__":
    app.run(port=port)
