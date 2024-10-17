from python_sound_input.listener import MicrophoneListener
from python_utilities.logger import setup_logging
import threading
from .speech_recognition import SpeechRecognition
from .config import SpeechRecognitionConfig
import queue
import logging

def listen():
    setup_logging(SpeechRecognitionConfig().default.log_level())

    logger = logging.getLogger(__name__)
    buffer_queue = queue.Queue()

    speech_recognition = SpeechRecognition()

    def transcribe():
        logger.info("Initializing transcription queue")
        while True:
            buffer = buffer_queue.get()
            print(speech_recognition.predict(buffer))

    listening_thread = threading.Thread(target=transcribe, name="listener")
    listening_thread.daemon = True
    listening_thread.start()

    listener = MicrophoneListener()
    listener.listen(buffer_queue.put)
