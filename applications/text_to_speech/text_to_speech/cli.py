from .text_to_speech import TextToSpeech
from python_utilities.logger import setup_logging
from .config import TextToSpeechConfig
import sounddevice


def speak():
    setup_logging(TextToSpeechConfig().default.log_level())

    tts = TextToSpeech()
    tts.setup_training()
    print(tts.train_embedding("../../resources/training_data/tracer.mp3", "../../resources/models/openvoice/embeddings", name="tracer"))
    tts.setup_prediction()

    wav, sampling_rate = tts.convert("Cheese is here. What do you want to say, love?")
    sounddevice.play(wav, sampling_rate)
    sounddevice.wait()