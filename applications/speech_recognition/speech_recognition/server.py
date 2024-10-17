from flask import Flask, request, make_response, jsonify
import numpy
import base64
from .config import SpeechRecognitionConfig
from .speech_recognition import SpeechRecognition
from python_utilities.logger import setup_logging
import logging
from datetime import datetime, timezone
import threading

port = SpeechRecognitionConfig().server.port()
host = SpeechRecognitionConfig().server.host()
debug = SpeechRecognitionConfig().server.debug()

app = Flask(__name__)
is_ready = False
is_healthy = True
model = None


def start_model():
    def start():
        global is_ready
        global is_healthy
        global model
        if model is None:
            try:
                model = SpeechRecognition()
                is_ready = True
                is_healthy = True
            except Exception as e:
                is_healthy = False
                is_ready = False
                app.logger.error("Model failed to initialize", exc_info=e)

    thread = threading.Thread(target=start, name="model")
    thread.daemon = True
    thread.start()


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health Check Endpoint
    ---
    Returns the health status of the application.

    Responses:
        200: Application is healthy.
        500: Application is unhealthy.
    """
    timestamp = (datetime.now(timezone.utc).isoformat() + "Z",)
    if not is_healthy:
        return jsonify({"status": "unhealthy", "timestamp": timestamp}), 500

    # Health check logic here (e.g., database connection, external service availability)
    health_status = {
        "status": "healthy",
        "timestamp": timestamp,
    }
    return jsonify(health_status), 200


@app.route("/status", methods=["GET"])
def readiness_check():
    """
    Readiness Check Endpoint
    ---
    Indicates if the application is ready to receive traffic.

    Responses:
        200: Application is ready.
        503: Application is not ready.
    """
    timestamp = (datetime.now(timezone.utc).isoformat() + "Z",)
    if not is_ready:
        return jsonify({"status": "not_ready", "timestamp": timestamp}), 503

    # Readiness check logic here (e.g., check if necessary services are available)
    readiness_status = {
        "status": "ready",
        "timestamp": timestamp,
    }
    return jsonify(readiness_status), 200


@app.route("/transcribe", methods=["POST"])
def transcribe():
    app.logger.info("Received request for transcription")

    while not is_ready:
        app.logger.warn("Awaiting app readiness")

    # Get the raw data from the request
    data = request.get_data()

    # Convert the bytes to a NumPy array (assuming the array is of dtype float32)
    array = numpy.frombuffer(base64.b64decode(data), dtype=numpy.float32)

    # Use the global model for prediction
    prediction = model.predict(array)

    response = make_response(jsonify(prediction), 200)
    response.mimetype = "text/plain"
    return response


def serve():
    setup_logging(SpeechRecognitionConfig().default.log_level())
    start_model()
    app.run(debug=debug, port=port, host=host)


if __name__ == "__main__":
    serve()
