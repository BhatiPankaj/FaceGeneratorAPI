# python main.py
# set FLASK_APP=main.py
# set FLASK_ENV=development
# flask run
# tensorflow==2.5.0


import base64
import gc
import io
# import flask as flask
import numpy as np
from flask_cors import CORS

from keras.preprocessing.image import img_to_array

from keras.models import load_model
from flask import Flask, jsonify, request
# import time
from keras.preprocessing.image import save_img
import tensorflow as tf

from PIL import Image
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
cors = CORS(app)

model = load_model('fully_trained.h5')
print(' Model Loaded ')


@app.route('/')
def home():
    return "Hello World"


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert('RGB')

    image = image.resize(target)
    image = img_to_array(image)

    image = (image - 127.5) / 127.5
    image = np.expand_dims(image, axis=0)

    return image


class Predict(Resource):
    def post(self):
        json_data = request.get_json()
        img_data = json_data['Image']

        image = base64.b64decode(str(img_data))

        img = Image.open(io.BytesIO(image))

        prepared_image = prepare_image(img, target=(256, 256))

        print(len(gc.get_objects()))

        preds = model.predict(prepared_image)

        outputFile = 'output.png'
        savePath = './output/'

        output = tf.reshape(preds, [256, 256, 3])

        output = (output + 1) / 2
        save_img(savePath + outputFile, img_to_array(output))

        imageNew = Image.open(savePath + outputFile)
        imageNew = imageNew.resize((50, 50))
        imageNew.save(savePath + "new_" + outputFile)

        with open(savePath + "new_" + outputFile, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read())

        gc.collect()
        print(len(gc.get_objects()))

        outputData = {
            'Image': str(encoded_string),
        }

        return outputData


api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)

# app.run(debug=True)
