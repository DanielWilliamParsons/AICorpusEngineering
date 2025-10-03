from AICorpusEngineering.mw_adverbs.extract import ExtractAdverbs
from pathlib import Path
import argparse
import os

def repo_root() -> Path:
    """Return the repository root."""
    # main.py is at src/AICorpusEngineering/mw_adverbs/
    return Path(__file__).resolve().parents[4]

def extract_features():
    """
    Learn the parameters of a logistic regression classifier for
    extracted multiword adverbs.
    """
    parser = argparse.ArgumentParser(description="Train a logistic classifier for multiword adverbs")
    parser.add_argument("corpus_dir", type=Path, help="Input the root directory of your corpus folder.")
    parser.add_argument("model_dir", type=Path, help="Input the directory where you will save the results of extracting features")
    args = parser.parse_args()

    corpus_path = args.corpus_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(corpus_path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            file_paths.append(full_path)
    
    for filepath in file_paths:
        extraction_tool = ExtractAdverbs(filepath, model_dir)
        extraction_tool.load_conll_file()
        extraction_tool.run_extraction()
