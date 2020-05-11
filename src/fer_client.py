"""
Client that talks to the fer_server.py

This is usually run on the Raspberry Pi, whereas the fer_server should be run on the server/desktop.
"""

import requestsimport cv2


class FERClient:
    def __init__(self, host="https://fer.cerbus.nl"):
        self.host = host

    def get_emotion(self, file):
        """Get emotion inside an image.

        Args:
            file: path to file on filesystem

        Returns:
            Dictionary with keys 'emotions', 'probabilities' and 'runtime'. Runtime has a dict with 'model', 'grayscale', and 'resize'. Runtime is given in milliseconds.
            Emotions/probabilities are ordered and sorted by descending confidence.
        """
        # preprocessing
        #importeren van het model: https://github.com/opencv/opencv/tree/master/data/haarcascades
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        image = cv2.imread( file )
        
        imageRGB = cv2.cvtColor( file, cv2.COLOR_BGR2RGB)
        imageCopy = imageRGB.copy()
        imageGray = cv2.cvtColor( imageRGB, cv2.COLOR_RGB2GRAY )

        face = faceCascade.detectMultiScale( imageGray, 1.25, 6)
        if not face:
            return
        else:
            #gezichten gevonden dus eerst croppen en dan resizen
            for (x, y, w, h) in face:
                cv2.rectangle(imageCopy, (x,y), (x+w, y+h), (255,0,0), 3)
                #croppen van gezicht
                faceCrop = imageGray[y:y+h, x:x+w]
                #resizen
                dim = (64, 64)
                resized = cv2.resize(faceCrop, dim, interpolation = methods[i])
                finishedImage = cv2.imencode('.png', resized)
        files = {"content": finishedImage.tostring()}
        r = requests.post(self.host + "/predict", files=files)
        return r.json()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("USAGE: fer_client.py [path_to_image]")
        sys.exit(1)

    client = FERClient()
    print(client.get_emotion(sys.argv[1]))
