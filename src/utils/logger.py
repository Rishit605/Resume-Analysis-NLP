import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file paths
APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "error.log")

# Formatters
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logger():
    """Setup and configure the logger with both file and console handlers"""
    # Create handlers
    app_handler = RotatingFileHandler(APP_LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    error_handler = RotatingFileHandler(ERROR_LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    # Create base logger
    logger = logging.getLogger("resume_api")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Add handlers
    logger.addHandler(app_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger

def get_log_status():
    """Returns information about the current logging setup"""
    log_files = {
        "app_log": APP_LOG_FILE,
        "error_log": ERROR_LOG_FILE
    }
    
    status = {}
    for log_type, log_file in log_files.items():
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            status[log_type] = {
                "file": log_file,
                "size_bytes": size,
                "size_mb": size / (1024 * 1024),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(log_file)).isoformat()
            }
        else:
            status[log_type] = {"status": "File not found"}
    
    return status

# Initialize logger
logger = setup_logger()

# Log the start of the logging system
logger.info("Logging system initialized")
