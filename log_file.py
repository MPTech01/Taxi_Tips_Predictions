import logging
import os 


# Define the log directory
log_dir = os.path.join(os.path.dirname(os.getcwd()), 'log_folder')

# Create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

def set_up_logs(name, log_file, level=logging.INFO):
    """
    Set up logging configuration for both file and console output.

    :param name: Name of the logger.
    :param log_file: Path to the log file where logs will be stored.
    :param level: Logging level (default is INFO).
    :return: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create file handler and set formatter
    file_name = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(filename=file_name)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Create stream handler and set formatter
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger 


def log_message(logger, level, message):
    """
    Log a message with a specified logging level.

    :param logger: Logger instance.
    :param level: The log level as a string (e.g., 'info', 'debug').
    :param message: The message to log.
    """
    level_map = {
        'info': logger.info, 
        'debug': logger.debug, 
        'warning': logger.warning, 
        'error': logger.error, 
        'critical': logger.critical
    }

    log_func = level_map.get(level.lower())  # Use get() to fetch the log function
    if log_func: 
        log_func(message)
    else:
        logger.error(f'Invalid log level: {level}, Message: {message}')
