from flask import Flask, request, jsonify, send_file
from http import HTTPStatus
import cv2
from tensorflow import keras
import numpy as np
from mtcnn import MTCNN

# img = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)
# detector.detect_faces(img)
import utils

app = Flask(__name__)

face_detector = MTCNN()


# utils functions here
def valid_format(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg', 'gif']


@app.route("/")
def hello():
    return "Welcome to my Vision & Cognitive Services Project"


@app.route("/test", methods=["GET"])
def test():
    img = cv2.cvtColor(cv2.imread("test.jpg"), cv2.COLOR_RGB2BGR)
    cv2.imshow("grr", img)
    cv2.waitKey(0)

    faces = face_detector.detect_faces(img)

    if len(faces) != 0:
        new_image = utils.detect_masks(input_image=img, bounding_boxes=faces)
        send_file()

    else:
        pass
        # todo: no faces detected

    return '''<b>Done</b>'''


@app.route("/upload", methods=["POST", "GET"])
def upload():

    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            return jsonify(status="error", message="No file detected"), HTTPStatus.BAD_REQUEST

        file = request.files.get("file")
        # if user does not select file, browser also
        # submit an empty part without filename

        if file.filename == '':
            return jsonify(status="error", message="No file detected"), HTTPStatus.BAD_REQUEST

        if not valid_format(file.filename):
            return jsonify(status="error", message="Extension not allowed"), HTTPStatus.BAD_REQUEST

        if file:
            file_bytes = file.read()
            np_img = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)

            img = utils.image_preprocessing(img)
            cv2.imshow("muller", img)
            cv2.waitKey(0)

            return jsonify(status="ok", result="muller"), HTTPStatus.OK
    else:
        # TODO: REMOVE THIS
        return '''
            <!doctype html>
            <title>Upload new File</title>
            <h1>Upload new File</h1>
            <form method="POST" enctype=multipart/form-data>
              <input type=file name=file>
              <input type=submit value=Upload>
            </form>
            '''


_PORT = 8080
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=_PORT)