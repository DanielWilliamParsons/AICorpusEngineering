import traceback
import sys

class ErrorHandler:
    """
    Handles errors across the whole project
    """
    def __init__(self, logger = None, fatal_exceptions = None):
        print("Initialize the error handling class.")
        """
        logger: an optional logger (NDJSONLogger)
        fatal_exceptions: a tuple of exceptions that should stop the program
        """
        self.logger = logger
        self.fatal_exceptions = fatal_exceptions or (RuntimeError,)

    def handle(self, exc: Exception, context: dict = None, fatal: bool = False):
        """
        Handle an exception: log it, and either re-raise or return a safe default
        """
        error_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }

        if self.logger:
            self.logger.log_records([{"error": error_info}])
        else:
            print("ERROR: ", error_info, file=sys.stderr)

        # Re-raise fatal errors
        if fatal or isinstance(exc, self.fatal_exceptions):
            raise

        # Otherwise, continue with a safe value
        return None

# Initialize a singleton that can be imported by any class
error_handler = ErrorHandler()