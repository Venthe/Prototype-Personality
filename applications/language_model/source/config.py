import configparser
import os


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.load_config()
        return cls._instance

    def load_config(self):
        config = configparser.ConfigParser()
        config.read("configuration.ini")
        if os.path.isfile("configuration-override.ini"):
            config.read("configuration-override.ini")
        self.config = config

    def model(self, key):
        return self.config["model"].get(key)

    def server(self, key):
        return self.config["server"].get(key)
