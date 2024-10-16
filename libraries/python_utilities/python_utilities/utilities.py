import time
from functools import wraps

def time_it(method):
    """Decorator to time the execution of a method."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()  # Record the start time
        result = method(self, *args, **kwargs)  # Call the original method
        end_time = time.time()    # Record the end time
        print(f'{method.__name__} execution took {end_time - start_time:.4f} seconds.')
        return result
    
    return wrapper