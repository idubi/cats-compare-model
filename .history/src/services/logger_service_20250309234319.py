import logging
import os

class LoggerService:
    _instance = None
    
    def __new__(cls, module_name=None, log_dir="LOGS"):
        if cls._instance is None:
            cls._instance = super(LoggerService, cls).__new__(cls)
            cls._instance._initialize(module_name, log_dir)
        return cls._instance
    
    def _initialize(self, module_name, log_dir):
        # Create LOGS directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(module_name if module_name else 'default')
        self.logger.setLevel(logging.DEBUG)
        
        # Create handlers
        log_file = f"{module_name}.log" if module_name else "default.log"
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        console_handler = logging.StreamHandler()
        
        # Create formatter
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)