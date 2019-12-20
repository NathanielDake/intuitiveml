def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode('utf-8')
    else:
        value = bytes_or_str
    return value


def is_numeric(val):
    """Determines if val can be cast to into integer."""
    try:
        int(float(val))
    except ValueError:
        return False
    return True
    
    
def cast_to_int_and_then_str(val):
    """
    Casts value to str, ensuring that any numerical value will be an integer. 
    
    Ex: [5, 6.0, 'a', 'b', 2] -> ['5', '6', 'a', 'b', '2']
    
    If we simply cast to string then 6.0 will be '6.0'.
    """
    if is_numeric(val):
        return str(int(float(val)))
    return val


def disable_warnings_temporarily(func):
    """Helper to disable warnings for specific functions (used mainly during testing of old functions)."""
    def inner(*args, **kwargs):
        
        import warnings
        warnings.filterwarnings("ignore")
        
        func(*args, **kwargs)
        
        warnings.filterwarnings("default")
    
    return inner