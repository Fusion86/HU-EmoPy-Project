"""
Client that talks to the fer_server.py

This is usually run on the Raspberry Pi, whereas the fer_server should be run on the server/desktop.
"""

import requests


class FERClient:
    def __init__(self, host="http://localhost:5000"):
        self.host = host

    def get_emotion(self, file):
        """Get emotion inside an image.

        Args:
            file: path to file on filesystem

        Returns:
            Dictionary with keys 'emotions', 'probabilities' and 'runtime'. Runtime has a dict with 'model', 'grayscale', and 'resize'. Runtime is given in milliseconds.
            Emotions/probabilities are ordered and sorted by descending confidence.
        """
        files = {"content": open(file, "rb")}
        r = requests.post(self.host + "/predict", files=files)
        return r.json()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("USAGE: fer_client.py [path_to_image]")
        sys.exit(1)

    client = FERClient()
    print(client.get_emotion(sys.argv[1]))
