import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = 'logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Create log file with timestamp
log_file = os.path.join(logs_dir, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def get_logger(name: str):
    """Get logger instance for a module"""
    return logging.getLogger(name)
