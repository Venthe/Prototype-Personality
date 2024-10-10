from flask import Flask, Response, request, jsonify
import logging
import threading
import queue

from config import Config
from language_model import LanguageModel

model = LanguageModel(Config().model("model_path"))
app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG or INFO based on what you want to see
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"An error occurred: {str(e)}")  # Log the error
    response = {"error": str(e), "message": "An unexpected error occurred."}
    return jsonify(response), 500


def generate_strings(messages):
    app.logger.debug("Requesting generation for %s", messages)
    model_generation_queue = queue.Queue()

    def generate():
        model.generate(
            messages,
            token_generated_callback=model_generation_queue.put,
            generation_done_callback=lambda w: model_generation_queue.put(None),
        )

    threading.Thread(target=generate).start()

    while True:
        item = model_generation_queue.get()
        app.logger.debug(f"Consuming word [{item}]")
        if item is None:
            app.logger.debug("Model finished generation")
            break
        # Server-Sent Events (SSE) protocol requires \n\n
        yield f"{item}\n\n".encode("utf8")


@app.route("/stream", methods=["POST"])
def stream():
    app.logger.info("Stream request received")
    # Get the JSON from request body
    data = request.get_json()

    # Convert "messages" from the JSON into a list of dicts
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    # Using chunked transfer encoding
    return Response(generate_strings(messages), content_type="text/event-stream")


@app.route("/health")
def health():
    app.logger.info("Health endpoint")
    return Response("OK", content_type="text/plain")


if __name__ == "__main__":
    app.run(port=Config().server("port"), threaded=True, debug=False)
