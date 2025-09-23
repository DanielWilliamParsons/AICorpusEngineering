import requests, json, re
from pathlib import Path
from datetime import datetime

class AdverbsAblationStudy:
    """
    This class monitors and edits agents
    in the adverbs ablation study
    """
    def __init__(self):
        print("intialize the class")