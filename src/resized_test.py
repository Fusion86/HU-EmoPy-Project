"""
Runs prediction on images which have been resized using different techniques.
Set `maaike_opencv_root` to the path containing a cloned https://github.com/MHovenkamp/Opencv
"""

# Settings
from collections import namedtuple
from fermodel import FERModel
import json
import cv2
import re
import glob
import os
maaike_opencv_root = "../../Opencv/Results/"
techniques = ["Area", "Cubic", "Lanczos4", "Linear", "Nearest"]


class ImgObj:
    def __init__(self, source, resized, age_group, gender, emotion):
        self.source = source  # source image path, we need this for the baseline
        self.resized = resized  # resized image path
        self.age_group = age_group  # y,m,o = young, middle aged, old
        self.gender = gender  # m,f = male, female
        self.emotion = emotion  # a,d,f,h,n,s = anger, disgust, fear, happiness, neutral, sad


FACESDB_NAME_MAP = {
    "a": "anger",
    "d": "disgust",
    "f": "fear",
    "h": "happiness",
    "n": "neutral",
    "s": "sadness"
}

if __name__ == "__main__":
    regex_str = r"([ymo])_([mf])_([adfhns])_([A-z])\.jpgGray\.png"

    model = FERModel("model.onnx")

    # Build testset
    testset = {}
    for technique in techniques:
        print("Building {} testset...".format(technique))
        testset[technique] = []
        abspath = os.path.join(os.path.abspath(maaike_opencv_root), technique)

        for f in glob.glob(os.path.join(abspath, "*_resized.png")):
            src = f.replace("_resized", "")
            m = re.search(regex_str, src)
            obj = ImgObj(src, f, m[1], m[2], m[3])
            testset[technique].append(obj)

    # Array containing values that match between FacesDB and FER+
    matching_keys = FACESDB_NAME_MAP.values()

    # Run testset
    results = {}
    for k in testset.keys():
        results[k] = {}
        total = 0
        correct = 0

        for obj in testset[k]:
            im = cv2.imread(obj.resized)
            detected = model.predict(im)['emotions']

            # Check if one of the top 2 emotions matches the expected emotion
            if detected[0] == FACESDB_NAME_MAP[obj.emotion] \
                or detected[1] == FACESDB_NAME_MAP[obj.emotion]:
                correct += 1

            # Only +1 the total counter if the emotion can match between the two datasets
            if detected[0] in matching_keys or detected[1] in matching_keys:
                total += 1

            print(obj.resized)
            print("Detected: {}, {}".format(detected[0], detected[1]))
            print("Classified as: " +
                  FACESDB_NAME_MAP[obj.emotion])
            print()

        results[k] = {
            "total": total,
            "correct": correct,
            "score": correct / total * 100
        }

    print(results)
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
