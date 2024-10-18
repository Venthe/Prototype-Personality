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
import numpy


class TextToSpeech:
    def __init__(self):

        self.logger = logging.getLogger(__name__)
        self.config = TextToSpeechConfig().openvoice
        self.cuda_enabled = self.config.use_gpu() and detect_cuda()
        # TODO: Allow the option to pick the proper device
        self.device = "cuda:0" if self.cuda_enabled else "cpu"

        self.text_to_speech = self.init_text_to_speech()
        self.tone_convert = self.init_tone_converter()

    def init_text_to_speech(self):
        tts_model = TTS(
            language=self.config.language_model().upper(), device=self.device
        )
        speaker_ids = tts_model.hps.data.spk2id
        self.logger.debug(f"Available speakers: {speaker_ids}")
        speaker_id = self.get_value_from_suffix(speaker_ids, self.config.speaker_key())

        def text_to_speech(text, speed=1.0):
            return (
                tts_model.tts_to_file(text, speed=speed, speaker_id=speaker_id),
                tts_model.hps.data.sampling_rate,
            )

        return text_to_speech

    def init_tone_converter(self):
        config_path = os.path.normpath(
            os.path.join(self.config.converter_path(), "config.json")
        )
        model_path = os.path.normpath(
            os.path.join(self.config.converter_path(), "checkpoint.pth")
        )
        tone_color_converter = ToneColorConverter(config_path, device=self.device)
        tone_color_converter.load_ckpt(model_path)

        embedding_checkpoint = self.init_embedding_checkpoint()
        speaker_model = self.init_speaker_model()

        def tone_convert(audio):
            return tone_color_converter.convert(
                audio_src_path=audio, src_se=speaker_model, tgt_se=embedding_checkpoint
            )

        return tone_convert

    def init_speaker_model(self):
        model_path = os.path.normpath(
            os.path.join(
                self.config.speaker_path(), f"{self.config.speaker_model()}.pth"
            )
        )
        return torch.load(model_path, map_location=torch.device(self.device))

    def init_embedding_checkpoint(self):
        embedding = f"{os.path.normpath(os.path.join(self.config.embedding_path(), self.config.embedding_model()))}.pth"
        embedding_checkpoint = torch.load(embedding).to(self.device)
        return embedding_checkpoint

    def get_value_from_suffix(self, data, text):
        text_lower = text.lower()
        for key in data.__dict__:
            suffix = key.split("-")[-1].lower()
            if text_lower == suffix:
                return data[key]
        raise KeyError(f"No matching key suffix found for '{text}'")

    def write(self, file, data, samplerate, subtype=None, endian=None, format=None,
            closefd=True):
        data = numpy.asarray(data)
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]
        with soundfile.SoundFile(file, 'w', samplerate, channels,
                    subtype, endian, format, closefd) as f:
            f.write(data)
            return f

    def to_soundfile(self, audio_data, sampling_rate=44100):
        audio_buffer = io.BytesIO()
        soundfile.write(audio_buffer, audio_data, sampling_rate, format="WAV")
        audio_buffer.seek(0)       

        return audio_buffer

    def convert(self, text, speed=1.0):
        audio_data, sampling_rate = self.text_to_speech(text, speed=speed)
        _audio_buffer = self.to_soundfile(audio_data, sampling_rate=sampling_rate)
        __audio_buffer = self.to_soundfile(self.tone_convert(_audio_buffer))
        read_file, _ = soundfile.read(__audio_buffer)
        return read_file, sampling_rate
