from flask import Flask, request, make_response, jsonify
import numpy
import base64
from .config import SpeechRecognitionConfig
from .speech_recognition import SpeechRecognition
from python_utilities.logger import setup_logging


port = SpeechRecognitionConfig().server.port()
host = SpeechRecognitionConfig().server.host()
debug = SpeechRecognitionConfig().server.debug()

model = SpeechRecognition()
app = Flask(__name__)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    app.logger.info("Received request for transcription")
    # Get the raw data from the request
    data = request.get_data()
    # print(data)

    # Convert the bytes to a NumPy array (assuming the array is of dtype float32)
    array = numpy.frombuffer(base64.b64decode(data), dtype=numpy.float32)
    app.logger.info(array)

    # Use the global model for prediction
    prediction = model.predict(array)

    response = make_response(jsonify(prediction), 200)
    response.mimetype = "text/plain"
    return response


def serve():
    setup_logging()
    app.run(debug=debug, port=port, host=host)


if __name__ == "__main__":
   serve()
