import warnings
        
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

import whisper
import os
import logging
import numpy
from .config import SpeechRecognitionConfig
from python_utilities.cuda import detect_cuda


class SpeechRecognition:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.config = SpeechRecognitionConfig().whisper
        self.cuda_enabled = self.config.use_gpu() and detect_cuda()
        # TODO: Allow the option to pick the proper device
        self.device = "cuda:0" if self.cuda_enabled else "cpu"

        # Create model directory
        model_directory = self.config.model_path()
        os.makedirs(model_directory, exist_ok=True)

        # Initialize device and model
        model_name = self.config.model_name()
        self.logger.info(f"Initiating model {os.path.normpath(os.path.join(model_directory, model_name))} on {self.device}")
        self.model = whisper.load_model(
            name=model_name,
            download_root=model_directory,
            device=self.device,
        )
        self.logger.info("Model initiated")

    # TODO: Add VAD to combat the hallucinations
    def predict(
        self,
        buffer,
        initial_prompt=None,
        temperature=None,
        hallucination_silence_threshold=None,
        language=None,
    ):
        if initial_prompt is None:
            initial_prompt = self.config.initial_prompt()
        if temperature is None:
            temperature = self.config.temperature()
        if hallucination_silence_threshold is None:
            hallucination_silence_threshold = self.config.hallucination_silence_threshold()
        if language is None:
            language = self.config.language()

        self.logger.debug("Transcribing...")

        transcription_output = self.model.transcribe(
            numpy.squeeze(self.normalize_audio(buffer)),
            fp16=self.cuda_enabled,
            initial_prompt=initial_prompt,
            temperature=temperature,
            hallucination_silence_threshold=hallucination_silence_threshold,
            language=language,
        )

        self.logger.debug(f"Transcription done: {transcription_output}")

        result = []
        for segment in transcription_output["segments"]:
            text = segment["text"].strip()
            no_speech_prob = segment["no_speech_prob"]
            if len(text) == 0:
                self.logger.warning("Speech not recognized")
                continue

            result.append({"text": text, "probability": 1 - no_speech_prob})
        return result

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
