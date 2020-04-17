# HU-EmotionDetection-Project

For a school project.
Tested on Python 3.7 but should also work on `Python >= 3.5`.

## Sources explained

### fermodel

Neural network that does emotion recognition.

### fer_server

Hosts a HTTP API on port 5000 which accepts images for which it returns the recognized emotions.
Usually runs on a server/desktop (as these have more processing power than a Raspberry Pi). See docstrings in fer_server.py for usage.

### fer_client

Client that connects to the fer_server. Usually runs on the Raspberry Pi.

## Setup

```sh
pip3 install -r requirements.txt # Server and tests
pip3 install -r requirements_client.txt # Client
```

## Example usage

```sh
# On server/desktop, start the FER server
python3 src/fer_server.py

# On Raspberry Pi, run the client
nano src/fer_client.py # Change host to point to the server
python3 src/fer_client.py images/keanu.jpg
```

## Todo

- Extract face from image (crop image to face)
- Add code to train the network (using CNTK)
- Allow grayscaled images (from fer_client) to skip the grayscaling step in fer_server because it already is in grayscale
