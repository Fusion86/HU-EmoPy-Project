import os
from EmoPy.src.fermodel import FERModel
from pkg_resources import resource_filename
import matplotlib.pyplot as plt

target_emotions = ["calm", "anger", "happiness"]
model = FERModel(target_emotions, verbose=True)

for root, dirs, files in os.walk(os.path.abspath("images")):
    for file in files:
        filePath = os.path.join(root, file)
        print("Processing {}...".format(filePath))
        prediction = model.predict(filePath)
        prediction.sort(key=lambda x: x[1], reverse=True)  # Sort by percentage

        # Loads image again + show GUI
        img = plt.imread(filePath)
        plt.imshow(img)

        # Write emotions
        infoStr = ""
        for x in prediction:
            infoStr += "{}: {:.2f}\n".format(x[0], x[1])

        plt.text(0, 0, infoStr)
        plt.show()
