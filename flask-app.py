import time
import base64
import os
import random
from flask import Flask, flash, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow import keras
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np
import random

UPLOAD_FOLDER = "C:\\Users\\yash1\\Desktop\\Masters - Fall term\\SW Project Mgmt\\Final-POC\\converted_keras"
application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# s3=boto3.client('s3')


@application.route('/')
def hello_world():
        return 'Hello World'

@application.route('/upload', methods=['POST'])
def upload():
    img = request.files.getlist('myfile')
    for x in img:
        filename=secure_filename(x.filename)
        path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
        x.save(path)
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Load the model
        model = keras.models.load_model("keras_model.h5", compile=False)

        # Load the labels
        class_names = open("labels.txt", 'r').readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
        image = Image.open(path).convert('RGB')

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

#turn the image into a numpy array
        image_array = np.asarray(image)
# Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
        data[0] = normalized_image_array

# run the inference
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        caloriesList = [89, 52, 47, 18, 131]
        calorieValue = caloriesList[index] + random.randint(-10,9)

        print('Class:', class_name, end='')
        print('Calories:', calorieValue, end='\n')
        print('Confidence score:', confidence_score)
        return jsonify(
                food_name=class_name[:-1],
                calories=calorieValue
        )

if __name__ == "__main__":
        application.run(host='0.0.0.0', port='8080')