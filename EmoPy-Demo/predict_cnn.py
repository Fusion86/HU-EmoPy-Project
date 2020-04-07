import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

# from EmoPy.src.neuralnets import ConvolutionalNN
from keras.models import model_from_json

target_dimensions = (64, 64)
channels = 1

model_file = "output/conv2d_model.json"
weights_file = "output/conv2d_weights.h5"
emotion_map_file = "output/conv2d_emotion_map.json"

model_json = open(model_file).read()
emotion_map = json.loads(open(emotion_map_file).read())

model = model_from_json(model_json)
model.load_weights(weights_file)

for root, dirs, files in os.walk(os.path.abspath("images")):
    for file in files:
        # Load image from disk
        image_file = os.path.join(root, file)
        print("Processing {}...".format(image_file))
        image_array = plt.imread(image_file)  # Reads file into numpy array

        # Convert BGR image to grayscale if it is not grayscale
        if len(image_array.shape) > 2:
            gray_image = cv2.cvtColor(image_array, code=cv2.COLOR_BGR2GRAY)

        # Resize image
        resized_image = cv2.resize(
            gray_image, target_dimensions, interpolation=cv2.INTER_LINEAR)

        # Convert image to some weird array, not sure what it does tho
        final_image = np.array([np.array([resized_image]).reshape(
            list(target_dimensions)+[channels])])

        # Do magic
        prediction = model.predict(final_image)

        # Convert magic output to undersetandable output
        emotions = []
        normalized_prediction = [x/sum(prediction) for x in prediction]
        for emotion in emotion_map.keys():
            emotions.append(
                (emotion, normalized_prediction[emotion_map[emotion]]*100))

        # Sort by percentage
        emotions.sort(key=lambda x: x[1], reverse=True)

        # Loads image again + show GUI
        img = plt.imread(image_file)
        plt.imshow(img)

        # Write emotions
        infoStr = ""
        for x in prediction:
            infoStr += "{}: {:.2f}\n".format(x[0], x[1])

        plt.text(0, 0, infoStr)
        plt.show()
