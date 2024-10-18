from python_config.config import Config
from python_config.utilities import parse_bool
from python_utilities.utilities import map_log_level



class Server():
    def __init__(self, data):
        self.data = data
    
    def port(self):
        return self.data.get("port")
    
    def host(self):
        return self.data.get("host")
    
    def debug(self):
        return parse_bool(self.data.get("debug"))
    

class Whisper():
    def __init__(self, data):
        self.data = data
    
    def model_path(self):
        return self.data.get("model_path")
    
    def model_name(self):
        return self.data.get("model_name")
    
    def initial_prompt(self):
        return self.data.get("initial_prompt")
    
    def use_gpu(self):
        return parse_bool(self.data.get("use_gpu"))
    
    def temperature(self):
        return float(self.data.get("temperature"))
    
    def hallucination_silence_threshold(self):
        return float(self.data.get("hallucination_silence_threshold"))
    
    def language(self):
        return self.data.get("language")
    

class Default():
    def __init__(self, data):
        self.data = data
    
    def log_level(self):
        return map_log_level(self.data.get("log_level"))
    
    def log_path(self):
        return self.data.get("log_path") if self.data.get("log_path") else None


class SpeechRecognitionConfig(Config):
    def __init__(self):
        super().__init__()
        self.whisper = Whisper(self.config["whisper"])
        self.server = Server(self.config["server"])
        self.default = Default(self.config["default"])
