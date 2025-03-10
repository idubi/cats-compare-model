import logging
import os
from datetime import datetime

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
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create handlers with timestamp in filename
        log_file = f"{module_name}_{timestamp}.log" if module_name else f"default_{timestamp}.log"
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        console_handler = logging.StreamHandler()
        
        # Create detailed formatter
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message, confidence=None, decision_reason=None):
        """Log info with optional confidence and decision explanation"""
        full_message = message
        if confidence is not None:
            full_message += f" [Confidence: {confidence:.2f}]"
        if decision_reason:
            full_message += f" [Reason: {decision_reason}]"
        self.logger.info(full_message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)