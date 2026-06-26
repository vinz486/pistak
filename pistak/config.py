import os
import json

class ConfigManager:
    def __init__(self, config_file="settings.json", models_dir="models"):
        self.config_file = config_file
        self.models_dir = models_dir
        self.settings = {
            "model_path": "",
            "device": "CPU",
            "port": "1234",
            "hf_token": ""
        }

    def load_settings(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    self.settings.update(json.load(f))
            except Exception:
                pass
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def save_settings(self):
        with open(self.config_file, "w") as f:
            json.dump(self.settings, f, indent=4)

    def get_local_models(self):
        if not os.path.exists(self.models_dir):
            return []
        return [(d, os.path.join(self.models_dir, d)) for d in os.listdir(self.models_dir) if os.path.isdir(os.path.join(self.models_dir, d))]
