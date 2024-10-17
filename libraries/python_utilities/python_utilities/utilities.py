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