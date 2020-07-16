import os
import time
from http import HTTPStatus
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from mtcnn import MTCNN
from tensorflow import keras

import config
import utils

app = Flask(__name__)
CORS(app)

start = time.time()
face_detector = MTCNN()
end = time.time() - start
print(f"it took {end}s to load MTCNN")



# utils functions here
def valid_format(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def process_image(np_img):
    original_img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)

    try:
        img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    except cv2.error:
        return jsonify(status="error",
                       message="that does not look like a valid image."), HTTPStatus.UNSUPPORTED_MEDIA_TYPE

    faces = face_detector.detect_faces(img)

    if len(faces) != 0:

        # cropped faces are already base64 encoded
        processed_img, cropped_faces = utils.detect_masks(input_image=img, bounding_boxes=faces)

        original_img = utils.cv_image_to_base64(input_image=original_img)
        processed_img = utils.cv_image_to_base64(input_image=processed_img)

        response = {
            "original": original_img,
            "processed": processed_img,
            "faces": cropped_faces
        }

        return jsonify(status="ok", result=response), HTTPStatus.OK

    else:
        response = "No faces were detected in that image :("
        return jsonify(status="error", message=response), HTTPStatus.BAD_REQUEST


@app.route("/")
def hello():
    return "Welcome to my Vision & Cognitive Services Project"


@app.route("/upload_url", methods=["POST"])
def upload_url():
    if "url" in request.form:

        url = request.form.get("url")
        if url == "":
            return jsonify(status="error", message="an empty url was given."), HTTPStatus.BAD_REQUEST

        schema = urlparse(url).scheme

        if schema not in ["http", "https"]:
            return jsonify(status="error", message="only http and https schemas are allowed."), HTTPStatus.BAD_REQUEST

        # retrieve image
        try:
            req = urlopen(Request(url, headers={"User-Agent": config.USER_AGENT}), timeout=5)
        except ValueError:
            return jsonify(status="error", message="that does not look like a valid URL."), HTTPStatus.BAD_REQUEST
        except URLError as e:
            print(str(e))
            return jsonify(status="error", message="that URL looks unreachable from here."), HTTPStatus.BAD_REQUEST

        # convert to numpy image
        np_img = np.asarray(bytearray(req.read()), dtype=np.uint8)

        # detect faces and masks
        response = process_image(np_img)

        return response
    else:
        return jsonify(status="error", message="required parameter is missing"), HTTPStatus.BAD_REQUEST


@app.route("/upload", methods=["POST"])
def upload():
    # check if the post request has the file part
    if "file" not in request.files:
        return jsonify(status="error", message="No file detected [1]"), HTTPStatus.BAD_REQUEST

    # retrieving file
    file = request.files.get("file")

    # checking if an empty file was submitted
    if file.filename == "":
        return jsonify(status="error", message="No file detected [2]"), HTTPStatus.BAD_REQUEST

    # checking if the extension is allowed
    if not valid_format(file.filename):
        return jsonify(status="error", message="extension not allowed"), HTTPStatus.BAD_REQUEST

    if file:
        # read image from storage
        file_bytes = file.read()

        # convert to numpy image
        np_img = np.frombuffer(file_bytes, np.uint8)

        # detect faces and masks
        response = process_image(np_img)

        return response

    else:
        return jsonify(status="error", message="No file detected [3]"), HTTPStatus.BAD_REQUEST


if __name__ == '__main__':

    # if PORT does not exist, set 8080
    _PORT = os.getenv("PORT", 80)
    app.run(host="0.0.0.0", port=_PORT)
