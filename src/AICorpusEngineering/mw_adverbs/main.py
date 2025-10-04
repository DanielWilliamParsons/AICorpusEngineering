from AICorpusEngineering.mw_adverbs.extract import ExtractAdverbs
from AICorpusEngineering.mw_adverbs.observe import ObserveAdverbs
from pathlib import Path
import argparse
import os

def repo_root() -> Path:
    """Return the repository root."""
    # main.py is at src/AICorpusEngineering/mw_adverbs/
    return Path(__file__).resolve().parents[4]

def observe_rules():
    """
    In order to extract candidate multi-word adverbs, we need to observe the rules
    that multi-word adverbs tend to follow.
    """
    parser = argparse.ArgumentParser(description="Observe rules followed by multiword adverbs")
    parser.add_argument("corpus_dir", type=Path, help="Input the root directory of the corpus you wish to observe")
    parser.add_argument("results_dir", type=Path, help="Input the folder name you wish to store the results of observations.")
    parser.add_argument("--results_fn", default="results.txt", help="type the name of the file you would like to store the results in.")
    parser.add_argument("--phrases", type=Path, help="The path to the text file containing the phrases you wish to observe.")
    args = parser.parse_args()

    corpus_path = args.corpus_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    results_path = results_dir / args.results_fn
    phrases_path = args.phrases.expanduser().resolve()

    # Create the phrases array
    phrases = []
    with open(phrases_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            phrases.append(line)

    # Get the corpus file names into an array
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(corpus_path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            file_paths.append(full_path)
    
    # For each file, observe the adverb phrase in the phrases list.
    for filepath in file_paths:
        extraction_tool = ObserveAdverbs(filepath, results_path)
        extraction_tool.load_conll_file()
        extraction_tool.extract_phrase_sentences(phrases)


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
