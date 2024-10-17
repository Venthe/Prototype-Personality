import os
import torch
from openvoice.api import ToneColorConverter
from melo.api import TTS
from datetime import datetime
import time
import io
from flask import Flask, request, send_file
import ffmpeg
import nltk


def setup():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('cmudict')

def convert_wav_to_mp3_memory(wav_file_path):
    """Convert a WAV file to MP3 and output to a memory buffer."""
    mem_file = io.BytesIO()
    process = (
        ffmpeg
        .input(wav_file_path)
        .output('pipe:', format='mp3')  # Output to pipe with MP3 format
        .run(capture_stdout=True, capture_stderr=True)
    )
    mem_file.write(process[0])
    mem_file.seek(0)

    return mem_file

config = {
    "device": "gpu",
    "openvoice_converter_path": "model/converter",
    "openvoice_base_speakers_path": "model/base_speakers",
    "embedding_file": "model/embeddings/se.pth",
    "speaker_language": "EN",
    "speed": 1.0,
    "temp_dir": "output"
}

def prepare_model():
    print("Preparing model")
    def get_speaker_key(tts_model):
        speaker_ids = tts_model.hps.data.spk2id
        speaker_key = "EN-Default"
        speaker_id = speaker_ids[speaker_key]
        return speaker_key.lower().replace("_", "-"), speaker_id

    openvoice_converter_checkpoint_config_file = f"{config['openvoice_converter_path']}/config.json"
    openvoice_converter_checkpoint_file = f"{config['openvoice_converter_path']}/checkpoint.pth"
    openvoice_embedding_file = f"{config['embedding_file']}"

    device = "cuda:0" if torch.cuda.is_available() and config["device"] == "gpu" else "cpu"
    
    os.makedirs(config["temp_dir"], exist_ok=True)

    tts_model = TTS(language=config['speaker_language'], device=device)

    speaker_key, speaker_id = get_speaker_key(tts_model)
    openvoice_speaker_checkpoint_file = f"{config['openvoice_base_speakers_path']}/ses/{speaker_key}.pth"

    speaker_checkpoint = torch.load(openvoice_speaker_checkpoint_file, map_location=torch.device(device))
    embedding_checkpoint = torch.load(openvoice_embedding_file).to(device)

    def color_convert(input, output):
        tone_color_converter = ToneColorConverter(
            openvoice_converter_checkpoint_config_file, device=device
        )
        tone_color_converter.load_ckpt(openvoice_converter_checkpoint_file)
        tone_color_converter.convert(
            audio_src_path=input,
            src_se=speaker_checkpoint,
            tgt_se=embedding_checkpoint,
            output_path=output,
        )
    print("Model prepared")
    return speaker_id, color_convert, tts_model