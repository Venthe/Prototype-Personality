import configparser
import os
import sys


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.load_config()
            cls._instance.override_file()
            cls._instance.override_environment_variables()
            cls._instance.override_arguments()
        return cls._instance

    def load_config(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), "configuration.ini"))

        self.config = config

    def override_file(self):
        override = self.config.get("default", "override_path", fallback="override.ini")
        if override != None and os.path.exists(override):
            self.config.read(override)

    def override_environment_variables(self):
        prefix = "CONFIG_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix to get section and key
                stripped_key = key[len(prefix) :]  # Remove prefix
                # Split the remaining key into section and key
                section, key = stripped_key.split("_", 1)

                self.override_value(section, key, value)

    def override_arguments(self):
        args = sys.argv[1:]

        # Process each argument
        for arg in args:
            if arg.startswith("--"):
                # Remove leading '--' and split into section and key
                stripped_key = arg[2:]  # Remove '--'

                if "=" in stripped_key:
                    # Split on '=' for key-value pairs
                    section_key, value = stripped_key.split("=", 1)
                else:
                    # No '=' means set value to None
                    section_key = stripped_key
                    value = None

                # Split section_key into section and key
                section, key = section_key.split("-", 1)

                self.override_value(section, key, value)

    def override_value(self, section, key, value):
        # Ensure the section exists in the config
        if not self.config.has_section(section):
            self.config.add_section(section)

        # Set the value in the config
        self.config.set(section, key, str(value) if value is not None else None)
