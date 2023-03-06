# Use a json file to store the configuration of the program
# The json file is used to store the configuration of the program
# The configuration file is stored in the same directory as the program


import json
import os

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = {}
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}

    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

if __name__ == '__main__':
    config = Config('config.json')
    config.set('name', 'Bob')
    config.set('age', 20)
    config.save_config()
    print(config.get('name'))
    print(config.get('age'))
