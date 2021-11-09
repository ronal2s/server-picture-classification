import os
import json
from flask import Flask, request, Response
from tensorflow.keras.models import load_model
from helpers.functions import read_and_prep_images, read_and_prep_image_from_url
# from helpers.functions import read_and_prep_images, read_and_prep_images2
app = Flask(__name__)

PATH_MODEL = './models/pictures_model.h5'
CLASSES = ['cellphone', 'digitalwatch', 'headphone',
           'laptop', 'speaker', 'tablet', 'television']

# image_paths = ['speaker1.png']
# image_paths = [
#     'https://m.media-amazon.com/images/I/61OvV27-44L._AC_SL1500_.jpg']
# image_data = read_and_prep_images2(image_paths)
# print('image_data', image_data)
new_model = load_model(PATH_MODEL)


@app.route("/predict", methods=["POST"])
def healthcheck():
    url = request.json['url']

    image_data = read_and_prep_image_from_url(url)
    predictions = new_model.predict(image_data)
    most_accurrate_prediction = predictions.argmax(axis=1)[0]
    most_accurrate_prediction = CLASSES[most_accurrate_prediction]
    response = {
        'prediction': most_accurrate_prediction
    }

    return Response(json.dumps(response), status=200, mimetype='application/json')


if __name__ == "__main__":
    app.run(host="0.0.0.0")
