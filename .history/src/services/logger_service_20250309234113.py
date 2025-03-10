import logging
from queue import Queue
from threading import Thread
import os

class LoggerService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        # Create LOGS directory if it doesn't exist
        os.makedirs("LOGS", exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('BluePrintLogger')
        self.logger.setLevel(logging.DEBUG)
        
        # Create handlers
        file_handler = logging.FileHandler('LOGS/dataset_creation.log')
        console_handler = logging.StreamHandler()
        
        # Create formatter
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
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