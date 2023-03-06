import json
from pathlib import Path

path = "/Users/apw804/dev-02/EnergyModels/KISS_06_config_template.json"

with open(path, "r") as f:
    config = json.load(f)

if __name__ == "__main__":
    print(config)