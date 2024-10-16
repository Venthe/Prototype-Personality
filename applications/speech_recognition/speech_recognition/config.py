from python_config.config import Config
from python_config.utilities import parse_bool


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


class SpeechRecognitionConfig(Config):
    def __init__(self):
        super().__init__()
        self.model = Model(self.config["model"])
        self.server = Server(self.config["server"])
