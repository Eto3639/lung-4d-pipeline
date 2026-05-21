import json
import os
from typing import Dict, Any

class ConfigManager:
    _instance = None

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config_path = config_path or os.path.join(os.getcwd(), 'config', 'thresholds.json')
            cls._instance.config = cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            # Fallback defaults if file missing
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return {}

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def get_thresholds(self, module_name: str) -> Dict[str, float]:
        return self.config.get(module_name, {})

    def reload(self):
        self.config = self._load_config()
