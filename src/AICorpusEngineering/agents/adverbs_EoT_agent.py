import requests, json, re
from pathlib import Path
from datetime import datetime

class AdverbsEoTAgent:
    """
    Runs the exclusion of thought
    tagging process
    """
    def __init__(self):
        print("initialize the class")