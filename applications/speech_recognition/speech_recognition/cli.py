from python_sound_input.listener import MicrophoneListener
from python_utilities.logger import setup_logging
import threading
from .speech_recognition import SpeechRecognition
from .config import SpeechRecognitionConfig
import queue
import logging
import time
from python_utilities.utilities import time_it

def listen():
    setup_logging(SpeechRecognitionConfig().default.log_level())

    logger = logging.getLogger(__name__)
    buffer_queue = queue.Queue()

    def receive_buffer(buffer):
        buffer_queue.put(buffer)

    def transcribe():
        logger.info("Initializing transcription queue")
        speech_recognition = SpeechRecognition()

        @time_it(logging.INFO)
        def predict(buffer):
            return speech_recognition.predict(buffer)

        while True:
            buffer = buffer_queue.get()
            print(predict(buffer))

    transcriber_thread = threading.Thread(target=transcribe, name="transcribe")
    transcriber_thread.daemon = True
    transcriber_thread.start()

    def listen():
        listener = MicrophoneListener()
        listener.listen(receive_buffer)

    listening_thread = threading.Thread(target=listen, name="listen")
    listening_thread.daemon = True
    listening_thread.start()

    while True:
        time.sleep(2)
