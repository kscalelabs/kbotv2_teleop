import logging
import sys
from dataclasses import dataclass


#* LOGGING
# Define ANSI escape sequences for colors
RESET = "\033[0m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"

class CustomFormatter(logging.Formatter):
    """Custom logging formatter to colorize log messages based on severity."""
    
    def format(self, record):
        # Customize the color based on the log level
        if record.levelno == logging.INFO:
            levelname_color = f"{BLUE}{record.levelname}{RESET}"
            record.msg = f"{BLUE}{record.msg}{RESET}"
        elif record.levelno == logging.WARNING:
            levelname_color = f"{RED}{record.levelname}{RESET}"
            record.msg = f"{RED}{record.msg}{RESET}"
        elif record.levelno == logging.DEBUG:
            levelname_color = f"\033[90m{record.levelname}{RESET}"  # Grey color for DEBUG
            record.msg = f"\033[90m{record.msg}{RESET}"
        else:
            levelname_color = record.levelname
        
        # Update the levelname in the record
        record.levelname = levelname_color
        return super().format(record)

def setup_logger(name, level=logging.INFO):
    """Set up and return a logger with the given name."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Set logging level
        logger.setLevel(level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = CustomFormatter('%(levelname)s: [%(name)s] %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger


