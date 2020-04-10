import os
import glob
import re
import json


from fermodel import FERModel
from collections import namedtuple


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
    "h": "happyness",
    "n": "neutral",
    "s": "sad"
}


if __name__ == "__main__":
    maaike_opencv_root = "../../Opencv/Results/"
    techniques = ["Area", "Cubic", "Lanczos4", "Linear", "Nearest"]

    regex_str = r"([ymo])_([mf])_([adfhns])_([A-z])\.jpgGray\.jpg"

    model = FERModel("model.onnx")

    # Build testset
    testset = {}
    for technique in techniques:
        print("Building {} testset...".format(technique))
        testset[technique] = []
        abspath = os.path.join(os.path.abspath(maaike_opencv_root), technique)

        for f in glob.glob(os.path.join(abspath, "*_resized.jpg")):
            src = f.replace("_resized", "")
            m = re.search(regex_str, src)
            obj = ImgObj(src, f, m[1], m[2], m[3])
            testset[technique].append(obj)

    # Run testset
    results = {}
    for k in testset.keys():
        results[k] = {}
        total = len(testset[k])
        correct = 0

        for obj in testset[k]:
            expected = model.predict(obj.source)['result'][0]
            actual = model.predict(obj.resized, True)['result'][0]

            if (expected == actual):
                correct += 1

            print(obj.source)
            print("Expected: " + expected)
            print("Actual: " + actual)
            print("Classified as (not important): " +
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
