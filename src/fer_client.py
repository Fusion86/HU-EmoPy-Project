"""
Client that talks to the fer_server.py

This is usually run on the Raspberry Pi, whereas the fer_server should be run on the server/desktop.
"""

import requests

# Settings
host = "http://localhost:5000"


def get_emotion(file):
    files = {"content": open(file, "rb")}
    r = requests.post(host + "/predict", files=files)
    return r.json()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("USAGE: fer_client.py [path_to_image]")
        sys.exit(1)

    print(get_emotion(sys.argv[1]))
