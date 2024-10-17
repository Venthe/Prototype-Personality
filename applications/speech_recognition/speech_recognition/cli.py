from python_sound_input.listener import MicrophoneListener
from python_utilities.logger import setup_logging
import threading
from .speech_recognition import SpeechRecognition
from .config import SpeechRecognitionConfig
import queue
import logging
import time


def transcribe(logger, buffer_queue):
    logger.info("Initializing transcription queue")
    speech_recognition = SpeechRecognition()

    while True:
        buffer = buffer_queue.get()
        prediction = speech_recognition.predict(buffer)
        if len(prediction) > 0:
            logger.info(f"Prediction: {prediction}")


def listen_to_input_device(buffer_queue):
    def receive_buffer(buffer):
        buffer_queue.put(buffer)

    listener = MicrophoneListener()
    listener.listen(receive_buffer)


def make_thread(target, name, args):
    thread = threading.Thread(target=target, name=name, args=args)
    thread.daemon = True
    thread.start()


def listen():
    setup_logging(SpeechRecognitionConfig().default.log_level())

    logger = logging.getLogger(__name__)
    buffer_queue = queue.Queue()

    make_thread(
        target=transcribe,
        name="transcribe",
        args=(
            logger,
            buffer_queue,
        ),
    )
    make_thread(target=listen_to_input_device, name="listen", args=(buffer_queue,))

    while True:
        time.sleep(2)
