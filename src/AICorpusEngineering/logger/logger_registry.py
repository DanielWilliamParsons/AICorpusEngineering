from typing import Optional

_logger_instance = None

def set_logger(logger):
    global _logger_instance
    _logger_instance = logger

def get_logger():
    global _logger_instance
    if _logger_instance is None:
        raise RuntimeError("Logger not initialized. Call set_logger() in main() first.")
    return _logger_instance