import logging
import queue
import time
import numpy
import sounddevice


class MicrophoneListener:
    def __init__(
        self,
        sample_rate=16000,
        block_size=3000,
        silence_threshold=0.005,
        input_device_index=None,
        split_silence_duration_seconds=1,
    ):
        self.logger = logging.getLogger(__name__)
        self.audio_queue = queue.Queue()

        self.sample_rate = sample_rate
        self.block_size = block_size
        self.silence_threshold = silence_threshold
        self.input_device_index = self.select_input_device(input_device_index)
        self.split_silence_duration_seconds = split_silence_duration_seconds

    def select_input_device(self, default_input_device_index=None):
        # List available audio input devices.
        self.logger.debug("Available audio input devices:")
        self.logger.debug(sounddevice.query_devices())
        input_device_index = (
            default_input_device_index
            if default_input_device_index is not None
            else sounddevice.default.device[0]
        )
        self.logger.debug(f"Selected microphone index: {input_device_index}")
        return input_device_index
    
    def get_sound_device_name(self, input_device_index):
        return sounddevice.query_devices()[input_device_index]['name']

    def listen(self, sound_recognized_callback=None):
        self.logger.info(f"Listening to microphone {self.get_sound_device_name(self.input_device_index)}")

        total_buffer = numpy.empty((0, 1), dtype=numpy.float32)
        current_buffer = numpy.empty((0, 1), dtype=numpy.float32)
        last_silence_time = time.time()
        speech_detected = False

        # TODO: Write a cleaner "clean" functions without so many variables
        with sounddevice.InputStream(
            samplerate=self.sample_rate,
            device=self.input_device_index,
            channels=1,
            blocksize=self.block_size,
            callback=self.audio_callback,
        ):
            while True:
                audio_data = self.audio_queue.get()
                # We use the current buffer to not append unnecessary silence to the samples
                current_buffer = numpy.append(current_buffer, audio_data)

                if not self.detect_silence_and_transcribe(audio_data):
                    self.logger.debug("Speech detected")
                    speech_detected = True
                    last_silence_time = time.time()
                    total_buffer = numpy.append(total_buffer, current_buffer)
                    current_buffer = numpy.empty((0, 1), dtype=numpy.float32)

                silence_duration = time.time() - last_silence_time
                if silence_duration <= self.split_silence_duration_seconds:
                    self.logger.debug(
                        f"Silence detected for duration {silence_duration}"
                    )
                    continue

                if not speech_detected:
                    # If the buffer is empty, reset the buffer
                    self.logger.debug("No speech detected in the buffer, resetting")
                    total_buffer = numpy.empty((0, 1), dtype=numpy.float32)
                    current_buffer = numpy.empty((0, 1), dtype=numpy.float32)
                    last_silence_time = time.time()
                    speech_detected = False
                    continue

                # Process the audio buffer, capturing additional preceding audio
                self.logger.info(
                    f"Queueing fragment of length {self.get_buffer_length(total_buffer)}s"
                )
                self.logger.debug(f"Queueing fragment of {len(total_buffer)} samples")

                # TODO: Split very long audio files into smaller chunks
                if sound_recognized_callback:
                    sound_recognized_callback(total_buffer.copy())

                total_buffer = numpy.empty((0, 1), dtype=numpy.float32)
                current_buffer = numpy.empty((0, 1), dtype=numpy.float32)
                last_silence_time = time.time()
                speech_detected = False

    def audio_callback(self, indata, frames, time, status):
        if status:
            self.logger.info(f"Audio stream status: {status}")
        self.audio_queue.put(indata.copy())

    def detect_silence_and_transcribe(self, buffer):
        energy = numpy.mean(numpy.abs(buffer))
        self.logger.debug(f"Energy: {energy}")
        return energy < self.silence_threshold

    def get_buffer_length(self, buffer):
        length_in_seconds = len(buffer) / self.sample_rate
        return length_in_seconds
