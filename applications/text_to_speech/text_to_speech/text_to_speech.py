import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`resume_download` is deprecated and will be removed in version 1.0.0.",
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="You are using `torch.load` with `weights_only=False`",
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`",
)

import os
import torch
import logging
from openvoice.api import ToneColorConverter
from melo.api import TTS
from .config import TextToSpeechConfig
from python_utilities.cuda import detect_cuda
import soundfile
import io
from openvoice import se_extractor
import shutil
from melo.download_utils import LANG_TO_HF_REPO_ID
import librosa


class TextToSpeech:
    def __init__(self):

        self.__logger = logging.getLogger(__name__)
        self.__config = TextToSpeechConfig().openvoice
        # TODO: Allow the option to pick the proper device
        self.__device = "cuda:0" if self.__config.use_gpu() and detect_cuda() else "cpu"

        self.__text_to_speech = self.__init_text_to_speech()
        self.__tone_convert = None
        self.__tone_converter = None
        self.__tone_converter_sampling_rate = None

    def setup_prediction(self):
        if self.__tone_converter is None:
            self.__init_tone_converter()
        if self.__tone_convert is None:
            self.__init_tone_convert()

    def setup_training(self):
        if self.__tone_converter is None:
            self.__init_tone_converter()

    def __init_tone_converter(self):
        config_path = os.path.normpath(
            os.path.join(self.__config.converter_path(), "config.json")
        )
        model_path = os.path.normpath(
            os.path.join(self.__config.converter_path(), "checkpoint.pth")
        )
        self.__tone_converter = ToneColorConverter(config_path, device=self.__device)
        self.__tone_converter.load_ckpt(model_path)
        self.__tone_converter_sampling_rate = (
            self.__tone_converter.hps.data.sampling_rate
        )

    def __init_tone_convert(self):
        embedding_checkpoint = self.__init_embedding_checkpoint()
        speaker_model = self.__init_speaker_model()

        def tone_convert(audio):
            return self.__tone_converter.convert(
                audio_src_path=audio, src_se=speaker_model, tgt_se=embedding_checkpoint
            )

        self.__tone_convert = tone_convert

    def train_embedding(
        self, reference_file, target_dir="./output", use_vad=True, clean=True
    ):
        if self.__tone_converter is None:
            raise BaseException("Tone converter is not yet created")

        self.__check_and_clean_directory(target_dir, clean)

        target_se, audio_name = se_extractor.get_se(
            audio_path=reference_file,
            vc_model=self.__tone_converter,
            vad=use_vad,
            target_dir=target_dir,
        )

        generated_directory = os.path.join(target_dir, audio_name)
        for item in os.listdir(generated_directory):
            if item == "wavs":
                continue
            item_path = os.path.join(target_dir, audio_name, item)
            shutil.move(item_path, target_dir)

        shutil.rmtree(generated_directory, ignore_errors=True)

        return os.path.join(target_dir, "se.pth")

    def __check_and_clean_directory(self, directory, clean=False):
        # Check if the directory exists
        if not os.path.exists(directory):
            raise ValueError("The specified directory does not exist.")

        # Check if the directory is empty
        if not os.listdir(directory):  # The directory is empty
            return True

        # The directory is not empty
        if clean:
            # Clean the directory by removing all contents
            shutil.rmtree(directory)
            os.makedirs(directory)  # Recreate the directory after cleaning
            return True
        else:
            raise ValueError("The directory is not empty.")

    def __init_text_to_speech(self):
        try:
            tts_model = TTS(
                language=self.__config.language_model().upper(), device=self.__device
            )
        except AssertionError as e:
            self.__logger.error(
                f"Language not supported, please use either of: {', '.join(list(LANG_TO_HF_REPO_ID.keys()))}"
            )
            raise

        speaker_ids = tts_model.hps.data.spk2id
        self.__logger.debug(f"Available speakers: {speaker_ids}")
        speaker_id = self.__get_value_from_suffix(
            speaker_ids, self.__config.speaker_key()
        )

        def text_to_speech(text, speed=1.0):
            return (
                tts_model.tts_to_file(text, speed=speed, speaker_id=speaker_id),
                tts_model.hps.data.sampling_rate,
            )

        return text_to_speech

    def __init_speaker_model(self):
        model_path = os.path.normpath(
            os.path.join(
                self.__config.speaker_path(), f"{self.__config.speaker_model()}.pth"
            )
        )
        return torch.load(model_path, map_location=torch.device(self.__device))

    def __init_embedding_checkpoint(self):
        embedding = f"{os.path.normpath(os.path.join(self.__config.embedding_path(), self.__config.embedding_model()))}.pth"
        embedding_checkpoint = torch.load(embedding).to(self.__device)
        return embedding_checkpoint

    def __get_value_from_suffix(self, data, text):
        text_lower = text.lower()
        for key in data.__dict__:
            suffix = key.split("-")[-1].lower()
            if text_lower == suffix:
                return data[key]
        raise KeyError(
            f"No matching key suffix found for '{text}', {', '.join(list(data.keys()))}"
        )

    def __to_soundfile(self, audio_data, sampling_rate=44100):
        audio_buffer = io.BytesIO()
        soundfile.write(audio_buffer, audio_data, sampling_rate, format="WAV")
        audio_buffer.seek(0)

        return audio_buffer, sampling_rate

    def __resample_audio(self, audio_buffer, target_sr=44100):
        audio_buffer.seek(0)
        audio_data, original_sampling_rate = soundfile.read(audio_buffer)

        # Resample the audio data
        resampled_audio = librosa.resample(
            audio_data, orig_sr=original_sampling_rate, target_sr=target_sr
        )

        return (self.__to_soundfile(resampled_audio, target_sr))

    def convert(self, text, speed=1.0):
        audio_data, sampling_rate = self.__text_to_speech(text, speed=speed)
        
        if self.__tone_convert is not None:
            result, sampling_rate = self.__to_soundfile(audio_data, sampling_rate=sampling_rate)
            result, sampling_rate = self.__resample_audio(
                result, self.__tone_converter_sampling_rate
            )
            result, sampling_rate = self.__to_soundfile(
                self.__tone_convert(result), self.__tone_converter_sampling_rate
            )
        else:
            result, sampling_rate = self.__to_soundfile(audio_data, sampling_rate=sampling_rate)
        sound_file, _ = soundfile.read(result)
        return (sound_file, sampling_rate)
