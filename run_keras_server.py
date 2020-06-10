# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2
import os
import flask
from flask import request, jsonify
from werkzeug import secure_filename

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)


@app.route('/')
def welcome():
   return "Client Signature Prediction"

@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        f = request.files["file"]
        f_name = secure_filename(f.filename)
        f.save(f_name)
        image = cv2.imread(f_name)
        image = cv2.resize(image, (64, 64))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        my_model = load_model('signature.model')
        my_lb = pickle.loads(open('lb.pickle', "rb").read())
        proba = my_model.predict(image)[0]
        idx = np.argmax(proba)
        label = my_lb.classes_[idx]
        filename = f_name[f_name.rfind(os.path.sep) + 1:]
        if proba[idx] < 0.8:
            label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, " Incorrect signature or considered forged..., try again")
        else:
            label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, "Correct Matching")

        result = "{}".format(label)
        res = jsonify(result)

        return res

if __name__ == "__main__":
    app.run()




















































