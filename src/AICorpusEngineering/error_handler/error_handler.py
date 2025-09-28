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
        self.logger = None # Don't get the logger on initialization
        self.fatal_exceptions = fatal_exceptions or (RuntimeError,)

    def handle(self, exc: Exception, context: dict = None, fatal: bool = False):
        """
        Handle an exception: log it, and either re-raise or return a safe default
        """
        # Get the logger if it is not yet set
        if self.logger is None:
            try:
                from AICorpusEngineering.logger.logger_registry import get_logger
                self.logger = get_logger()
            except RuntimeError:
                self.logger = None

        error_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }

        if self.logger:
            self.logger.log_error([{"error": error_info}])
        else:
            print("ERROR: ", error_info, file=sys.stderr)

        # Re-raise fatal errors
        if fatal or isinstance(exc, self.fatal_exceptions):
            raise

        # Otherwise, continue with a safe value
        return None

# Initialize a singleton that can be imported by any class
# TODO: add the logger
error_handler = ErrorHandler()