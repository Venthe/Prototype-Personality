from .text_to_speech import TextToSpeech
from python_utilities.logger import setup_logging
from .config import TextToSpeechConfig
import sounddevice


def speak():
    setup_logging(TextToSpeechConfig().default.log_level())

    tts = TextToSpeech()
    wav, sampling_rate = tts.convert("text")
    sounddevice.play(wav, sampling_rate)
    sounddevice.wait()