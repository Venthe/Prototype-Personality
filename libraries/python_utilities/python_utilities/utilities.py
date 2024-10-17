import time
from functools import wraps
import logging

def time_it(log_level=logging.DEBUG):
    def decorator(method):
        """Decorator to time the execution of a method."""
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            logger = logging.getLogger(__name__)
            start_time = time.time()  # Record the start time
            result = method(self, *args, **kwargs)  # Call the original method
            end_time = time.time()    # Record the end time
            logger.log(log_level, f'{method.__name__} execution took {end_time - start_time:.4f} seconds.')
            return result
        
        return wrapper
    return decorator

def map_log_level(log_level_str):
    """
    Map a string representation of a log level to the corresponding logging level.
    
    Args:
        log_level_str (str): The string representation of the log level (e.g., 'DEBUG', 'info').
    
    Returns:
        int: The corresponding logging level as an integer.
    
    Raises:
        ValueError: If the input string does not correspond to a valid log level.
    """
    # Create a mapping of string representations to logging levels
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }

    # Normalize the input string to upper case
    normalized_level = log_level_str.upper()

    # Retrieve and return the corresponding logging level
    if normalized_level in log_levels:
        return log_levels[normalized_level]
    else:
        raise ValueError(f"Invalid log level: '{log_level_str}'")