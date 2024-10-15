from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import torch
import os
import shutil

config = {
    "device": "gpu",
    "checkpoint_path": "model/converter",
    "reference_file": "training_data/reference.mp3",
    "output_path": "output",
    "use_vad": False,
    "output_name": "se",
}
device = "cuda:0" if torch.cuda.is_available() and config["device"] == "gpu" else "cpu"
checkpoint_configuration = os.path.join(config["checkpoint_path"], "config.json")
checkpoint_file = os.path.join(config["checkpoint_path"], "checkpoint.pth")

print("Establishing a tone color converter")
tone_color_converter = ToneColorConverter(
    config_path=checkpoint_configuration, device=device
)
tone_color_converter.load_ckpt(ckpt_path=checkpoint_file)
print("Converter tone color established")

print("Preparing tone color embedding")
target_se, audio_name = se_extractor.get_se(
    audio_path=config["reference_file"],
    vc_model=tone_color_converter,
    vad=config["use_vad"],
    target_dir=config["output_path"],
)
os.makedirs(config["output_path"], exist_ok=True)
os.rename(
    src=os.path.join(config["output_path"], audio_name, "se.pth"),
    dst=os.path.join(config["output_path"], f"{config['output_name']}.pth"),
)
shutil.rmtree(os.path.join(config["output_path"], audio_name), ignore_errors=True)
print(f"Tone color embedding created. {audio_name}")
