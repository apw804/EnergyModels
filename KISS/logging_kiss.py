import logging
from pathlib import Path

from utils_kiss import get_timestamp

def get_logger(logger_name, log_dir, log_level=logging.INFO):
    """
    Create and return a logger object.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Set up file handler to write all logging messages to the same file
    log_file_path = Path(log_dir) / f'{logger_name}_{get_timestamp()}.log'
    handler = logging.FileHandler(log_file_path, mode="a")
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set up stream handler to write logging messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger