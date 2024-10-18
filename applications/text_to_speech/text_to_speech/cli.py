from .text_to_speech import TextToSpeech
from python_utilities.logger import setup_logging
from .config import TextToSpeechConfig
import sounddevice


def speak():
    setup_logging(TextToSpeechConfig().default.log_level())

    tts = TextToSpeech()
    tts.init_tone_convert()
    audio_name, target_se = tts.train_embedding("../../resources/training_data/reference.mp3", "output/")

    #wav, sampling_rate = tts.convert("Cheese is here")
    #sounddevice.play(wav, sampling_rate)
    #sounddevice.wait()
    print(audio_name, target_se)