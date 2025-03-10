import json
import os

class ConfigManager:
    """Manages all configuration settings for augmentations, logging, and paths."""
    
    _instance = None  # Singleton instance
    
    def __new__(cls, config_file="config.json"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.load_config(config_file)
        return cls._instance

    def load_config(self, config_file):
        """Loads configuration settings from a JSON file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' not found.")
        
        with open(config_file, "r") as file:
            self.config = json.load(file)

        self.paths = self.config.get("paths", {})
        self.augmentations = self.config.get("augmentations", {})
        self.logging = self.config.get("logging", {})
        
        print("✅ Config Loaded Successfully!")

    def get_path(self, key):
        """Retrieves a file path from the config."""
        return self.paths.get(key, "")

    def is_augmentation_enabled(self, aug_type):
        """Checks if a specific augmentation is enabled."""
        return self.augmentations.get(aug_type, {}).get("enabled", False)

    def get_augmentation_probability(self, aug_type):
        """Gets the probability weight for an augmentation."""
        return self.augmentations.get(aug_type, {}).get("probability", 0)

    def should_log_to_console(self):
        """Checks if console logging is enabled."""
        return self.logging.get("console", True)

    def should_log_to_file(self):
        """Checks if logging to a file is enabled."""
        return self.logging.get("file", True)

    def get_log_file(self):
        """Gets the log file name."""
        return self.logging.get("log_file", "augmentations.json")
