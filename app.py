import os
import json
from flask import Flask, request, Response
from tensorflow.keras.models import load_model
from helpers.functions import read_and_prep_images, read_and_prep_image_from_url
# from helpers.functions import read_and_prep_images, read_and_prep_images2
app = Flask(__name__)

# PATH_MODEL = './models/pictures_model.h5'
PATH_MODEL = './models/model_trainingv3.h5'
# CLASSES = ['cellphone', 'digitalwatch', 'headphone',
#            'laptop', 'speaker', 'tablet', 'television'] #v2
CLASSES = ['cellphone', 'digitalwatch', 'headphone',
           'laptop', 'television'] #v3

# image_paths = ['test.png']
# image_data = read_and_prep_images(image_paths)

new_model = load_model(PATH_MODEL)


@app.route("/", methods=["GET"])
def test():
    return Response("Proyecto Final Server", status=200)

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
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

    # app.run(threaded=True, port=5000)
    # app.run(host="0.0.0.0")
