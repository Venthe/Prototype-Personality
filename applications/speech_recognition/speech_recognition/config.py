from python_config.config import Config
from python_config.utilities import parse_bool
from python_utilities.utilities import map_log_level



class Server():
    def __init__(self, server):
        self.server = server
    
    def port(self):
        return self.server.get("port")
    
    def host(self):
        return self.server.get("host")
    
    def debug(self):
        return parse_bool(self.server.get("debug"))
    

class Model():
    def __init__(self, model):
        self.model = model
    
    def model_path(self):
        return self.model.get("model_path")
    
    def model_name(self):
        return self.model.get("model_name")
    
    def initial_prompt(self):
        return self.model.get("initial_prompt")
    
    def use_gpu(self):
        return parse_bool(self.model.get("use_gpu"))
    
    def temperature(self):
        return float(self.model.get("temperature"))
    
    def hallucination_silence_threshold(self):
        return float(self.model.get("hallucination_silence_threshold"))
    
    def language(self):
        return self.model.get("language")
    

class Default():
    def __init__(self, model):
        self.model = model
    
    def log_level(self):
        return map_log_level(self.model.get("log_level"))
    
    def log_path(self):
        return self.model.get("log_path") if self.model.get("log_path") else None


class SpeechRecognitionConfig(Config):
    def __init__(self):
        super().__init__()
        self.model = Model(self.config["model"])
        self.server = Server(self.config["server"])
        self.default = Default(self.config["default"])
