from python_config.config import Config
from python_config.utilities import parse_bool
import logging

def map_log_level(log_level_str):
    """
    Map a string representation of a log level to the corresponding logging level.
    
    Args:
        log_level_str (str): The string representation of the log level (e.g., 'DEBUG', 'info').
    
    Returns:
        int: The corresponding logging level as an integer.
    
    Raises:
        ValueError: If the input string does not correspond to a valid log level.
    """
    # Create a mapping of string representations to logging levels
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }

    # Normalize the input string to upper case
    normalized_level = log_level_str.upper()

    # Retrieve and return the corresponding logging level
    if normalized_level in log_levels:
        return log_levels[normalized_level]
    else:
        raise ValueError(f"Invalid log level: '{log_level_str}'")


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
