def parse_bool(value):
    """
    Parse a string value into a boolean (True or False).
    
    Parameters:
        value (str): The string to parse.
        
    Returns:
        bool: The corresponding boolean value.
        
    Raises:
        ValueError: If the string is not a valid boolean representation.
    """
    if isinstance(value, str):
        value = value.strip().lower()  # Normalize the string
        if value == 'true':
            return True
        elif value == 'false':
            return False
    
    raise ValueError(f"Invalid boolean string: '{value}'")