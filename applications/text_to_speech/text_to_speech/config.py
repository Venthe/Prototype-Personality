from python_config.config import Config
from python_config.utilities import parse_bool
from python_utilities.utilities import map_log_level


class OpenVoice:
    def __init__(self, data):
        self.data = data

    def use_gpu(self):
        return parse_bool(self.data.get("use_gpu"))
    
    def converter_path(self):
        return self.data.get("converter_path")
    
    def speaker_path(self):
        return self.data.get("speaker_path")
    
    def speaker_model(self):
        return self.data.get("speaker_model")
    
    def embedding_path(self):
        return self.data.get("embedding_path")
    
    def embedding_model(self):
        return self.data.get("embedding_model")
    
    def language_model(self):
        return self.data.get("language_model")
    
    def speaker_key(self):
        return self.data.get("speaker_key")
    

class Default():
    def __init__(self, data):
        self.data = data
    
    def log_level(self):
        return map_log_level(self.data.get("log_level"))
    
    def log_path(self):
        return self.data.get("log_path") if self.data.get("log_path") else None

class TextToSpeechConfig(Config):
    def __init__(self):
        super().__init__()
        self.openvoice = OpenVoice(self.config["openvoice"])
        self.default = Default(self.config["default"])
