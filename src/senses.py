import logging
import threading
import queue
import time
import numpy
import sounddevice
import httpx
import base64
import json
from box import Box

logger = logging.getLogger(__name__)

def process_senses(knowledge_message_callback = None):
    logger.info("Starting sense processing thread")

    transcribe_callback = setup_transcribe(knowledge_message_callback)
    setup_listen(transcribe_callback)

def setup_transcribe(knowledge_message_callback = None):
    transcription_queue = queue.Queue()
    def transcribe():
        dlq = None
        while True:
            data = None
            if dlq is not None:
                data = dlq
                dlq = None
            else:
                data = transcription_queue.get()
            try:
                response = httpx.post("http://localhost:5000/transcribe", data=base64.b64encode(data.tobytes()))
                response.raise_for_status()

                if knowledge_message_callback:
                    knowledge_message_callback(Box({
                        "type": "speech",
                        "data": json.loads(response.text)
                    }))
            except Exception as e:
                print(e)
                dlq = data
                time.sleep(1)
    listening_thread = threading.Thread(target=transcribe, name="Transcriber")
    listening_thread.daemon = True
    listening_thread.start()

    return transcription_queue.put

def setup_listen(audio_buffer_callback):
    def listen():
        listener = MicrophoneListener(audio_buffer_callback)
        listener.listen()
    listening_thread = threading.Thread(target=listen, name="Microphone-Listener")
    listening_thread.daemon = True
    listening_thread.start()

class MicrophoneListener:
    def __init__(self, audio_buffer_callback = None):
        self.audio_queue = queue.Queue()
        self.config = {
            "sample_rate": 16000,
            "block_size": 3000,
            "silence_threshold": 0.005,
            "default_mic_index": None,
            "min_silence_duration": 1,
        }
        self.mic_index = self.list_microphones()
        self.audio_buffer_callback = audio_buffer_callback

    def listen(self):
        logger.debug(f"Listening to microphone {self.mic_index}")

        total_buffer = numpy.empty((0, 1), dtype=numpy.float32)
        current_buffer = numpy.empty((0, 1), dtype=numpy.float32)
        last_silence_time = time.time()
        speech_detected = False

        # TODO: Write a cleaner "clean" functions without so many variables
        with sounddevice.InputStream(
            samplerate=self.config["sample_rate"],
            device=self.mic_index,
            channels=1,
            blocksize=self.config["block_size"],
            callback=self.audio_callback,
        ):
            while True:
                audio_data = self.audio_queue.get()
                # We use the current buffer to not append unnecessary silence to the samples
                current_buffer = numpy.append(current_buffer, audio_data)

                if not self.detect_silence_and_transcribe(audio_data):
                    logger.debug("Speech detected")
                    speech_detected = True
                    last_silence_time = time.time()
                    total_buffer = numpy.append(total_buffer, current_buffer)
                    current_buffer = numpy.empty((0, 1), dtype=numpy.float32)

                silence_duration = time.time() - last_silence_time
                if silence_duration <= self.config["min_silence_duration"]:
                    logger.debug(
                        f"Silence detected for duration {silence_duration}"
                    )
                    continue

                if not speech_detected:
                    """ If the buffer is empty, reset the buffer """
                    logger.debug("No speech detected in the buffer, resetting")
                    total_buffer = numpy.empty((0, 1), dtype=numpy.float32)
                    current_buffer = numpy.empty((0, 1), dtype=numpy.float32)
                    last_silence_time = time.time()
                    speech_detected = False
                    continue

                # Process the audio buffer, capturing additional preceding audio
                logger.info(
                    f"Queueing fragment of length {self.get_buffer_length(total_buffer)}s"
                )
                logger.debug(f"Queueing fragment of {len(total_buffer)} samples")

                # TODO: Split very long audio files into smaller chunks
                if self.audio_buffer_callback:
                    self.audio_buffer_callback(total_buffer.copy())

                total_buffer = numpy.empty((0, 1), dtype=numpy.float32)
                current_buffer = numpy.empty((0, 1), dtype=numpy.float32)
                last_silence_time = time.time()
                speech_detected = False

    def list_microphones(self):
        """List available audio input devices."""
        logger.debug("Available audio input devices:")
        logger.debug(sounddevice.query_devices())
        mic_index = (
            self.config["default_mic_index"]
            if self.config["default_mic_index"] is not None
            else sounddevice.default.device[0]
        )
        logger.debug(f"Selected microphone index: {mic_index}")
        return mic_index

    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.info(f"Audio stream status: {status}")
        self.audio_queue.put(indata.copy())

    def detect_silence_and_transcribe(self, buffer):
        energy = numpy.mean(numpy.abs(buffer))
        logger.debug(f"Energy: {energy}")
        return energy < self.config["silence_threshold"]

    def get_buffer_length(self, buffer):
        # Calculate length in seconds
        length_in_seconds = len(buffer) / self.config["sample_rate"]
        return length_in_seconds