import whisper
import os
import logging
import warnings
import numpy
from .config import SpeechRecognitionConfig
from python_utilities.utilities import time_it
from python_utilities.cuda import detect_cuda


class SpeechRecognition:
    @time_it
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initiating model")
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r"^.*torch\.load.*weights_only=False.*$",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"^.*1Torch was not compiled with flash attention.*$",
        )

        # Configuration Variables
        self.config = {
            "cuda_enabled": detect_cuda(),
            "model_name": "medium.en",
            "model_directory": SpeechRecognitionConfig().model.model_path(),
            "initial_prompt": "You are a british speaker, please transcribe this into English for me.",
        }

        # Create model directory
        os.makedirs(self.config["model_directory"], exist_ok=True)

        # Initialize device and model
        self.device = "cuda:0" if self.config["cuda_enabled"] else "cpu"

        self.model = whisper.load_model(
            name=self.config["model_name"],
            download_root=self.config["model_directory"],
            device=self.device,
        )
        self.logger.info("Model initiated")

    # TODO: Add VAD to combat the hallucinations
    @time_it
    def predict(self, buffer):
        self.logger.info(f"Starting with the device {self.device}")
        self.logger.info("Transcribing...")
        fp16 = self.config["cuda_enabled"]

        sample = self.normalize_audio(
            self.pad_buffer(buffer, noise=False, duration=None)
        )
        result = self.model.transcribe(
            sample,
            fp16=fp16,
            initial_prompt=self.config["initial_prompt"],
            # TODO: Extract variable to config
            temperature=0,
            # TODO: Extract variable to config
            hallucination_silence_threshold=1,
            # TODO: Extract variable to config
            language="en",
        )

        self.logger.debug(f"{result}")

        r = []
        for segment in result["segments"]:
            # TODO: Extract variable to config
            text = segment["text"].strip()
            no_speech_prob = segment["no_speech_prob"]
            if len(text) == 0:
                self.logger.info("Speech not recognized")
                continue

            r.append({"text": text, "probability": 1 - no_speech_prob})
        return r

    def pad_buffer(self, buffer, duration=0.1, noise=True):
        if duration == None:
            return buffer

        # Create silence buffer
        padding = (
            numpy.zeros(int(self.config["sample_rate"] * duration), dtype=numpy.float32)
            if not noise
            # TODO: Extract variable to config
            else (numpy.random.randn(self.config["sample_rate"]) * 0.02).astype(
                numpy.float32
            )
        )

        # Concatenate silence before and after the original buffer
        final_buffer = numpy.concatenate(
            (padding, buffer.astype(numpy.float32), padding)
        )
        return final_buffer

    def normalize_audio(self, buffer):
        # Calculate the maximum absolute value in the buffer
        max_value = numpy.max(numpy.abs(buffer))

        # Avoid division by zero
        if max_value > 0:
            normalized_buffer = buffer / max_value
        else:
            """If max_value is zero, return the original buffer"""
            normalized_buffer = buffer

        return normalized_buffer
