
class ErrorHandler:
    """
    Handles errors across the whole project
    """
    def __init__(self):
        print("Initialize the error handling class.")


# Initialize a singleton that can be imported by any class
error_handler = ErrorHandler()