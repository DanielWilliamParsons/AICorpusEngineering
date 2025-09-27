from pathlib import Path
import argparse
import os
import importlib.resources as resources
from datetime import datetime

def repo_rooe() -> Path:
    """Return the repo root."""
    return Path(__file__).resolve.parents[4]

def load_data(path_to_data):
    """
    Load the sentences and read into memory
    """
    sentences = []
    with path_to_data.open("r", encoding="utf-8") as f:
        for line in f:
            sentences.append(line)
    print(sentences)


def main():
    """
    This is the endpoint for the manual tagger.
    This will only work if the sentences for tagging have been sampled from the corpus, extracted and saved to file.
    The output of this work is a set of gold standard tagged sentences, tagged by a human.
    Each sentence is presented and the human must select the correct set of semantic tags
    """
    parser = argparse.ArgumentParser(description="Run the manual tagger")
    parser.add_argument("input_dir", type=Path, help="Input the name of the directory where the sampled sentences are stored")
    parser.add_argument("file_name", help="Input the name of the file where the sampled sentences are saved.")

    args = parser.parse_args()
    path_to_data = args.input_dir / args.file_name
    path_to_data = path_to_data.expanduser().resolve()

    if not path_to_data.exists():
        raise FileNotFoundError(f"Path to data not found: {path_to_data}")
    
    load_data(path_to_data)
    