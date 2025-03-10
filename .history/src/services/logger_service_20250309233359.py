import logging
import asyncio
from queue import Queue
from threading import Thread
import os
from datetime import datetime

class AsyncLoggerService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AsyncLoggerService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.log_queue = Queue()
        self.logger = self._setup_logger()
        self.running = True
        
        # Start worker thread for processing logs
        self.worker_thread = Thread(target=self._process_logs)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def _setup_logger(self):
        # Create LOGS directory if it doesn't exist
        os.makedirs("LOGS", exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('BluePrintLogger')
        logger.setLevel(logging.DEBUG)
        
        # Create handlers
        file_handler = logging.FileHandler('LOGS/dataset_creation.log')
        console_handler = logging.StreamHandler()
        
        # Create formatter
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _process_logs(self):
        while self.running:
            try:
                if not self.log_queue.empty():
                    level, message = self.log_queue.get()
                    if level == logging.DEBUG:
                        self.logger.debug(message)
                    elif level == logging.INFO:
                        self.logger.info(message)
                    elif level == logging.WARNING:
                        self.logger.warning(message)
                    elif level == logging.ERROR:
                        self.logger.error(message)
                    elif level == logging.CRITICAL:
                        self.logger.critical(message)
                else:
                    # Small sleep to prevent CPU spinning
                    asyncio.run(asyncio.sleep(0.1))
            except Exception as e:
                print(f"Error processing log: {e}")
    
    async def log(self, level, message):
        self.log_queue.put((level, message))
    
    async def debug(self, message):
        await self.log(logging.DEBUG, message)
    
    async def info(self, message):
        await self.log(logging.INFO, message)
    
    async def warning(self, message):
        await self.log(logging.WARNING, message)
    
    async def error(self, message):
        await self.log(logging.ERROR, message)
    
    async def critical(self, message):
        await self.log(logging.CRITICAL, message)
    
    def shutdown(self):
        self.running = False
        self.worker_thread.join()