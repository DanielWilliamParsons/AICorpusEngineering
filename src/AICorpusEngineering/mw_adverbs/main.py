from AICorpusEngineering.mw_adverbs.extract import ExtractAdverbs
from AICorpusEngineering.mw_adverbs.observe import ObserveAdverbs
from pathlib import Path
import argparse
import os
from collections import defaultdict
import random
import json

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
    parser.add_argument("results_path", type=Path, help="Input the path to the file, including file name, where you wish to store the results of observations.")
    parser.add_argument("--phrases", type=str, help="The value of the generalized POS that you wish to examine, e.g., 'ADV * ADV' for adverb phrases beginning with an adverb and ending with an adverb.")
    args = parser.parse_args()

    corpus_path = args.corpus_dir.expanduser().resolve()
    results_path = args.results_path.expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)


    script_dir = Path(__file__).resolve().parent
    multiword_adverbs_file_path = script_dir / "multiword_adverbs_generalized.ndjson"

    # Create the phrases array
    phrases = None
    with open(multiword_adverbs_file_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            # Select the generalized adverb pattern
            json_line = json.loads(line)
            if json_line["POS_GEN"] == args.phrases:
                phrases = json_line["Phrases"]
    if phrases is None:
        print("There are no phrases with your selected adverb pattern")
    
    all_phrases = phrases.split(", ")
    all_phrases_unique = list(set(all_phrases))
    groups = defaultdict(list)
    for phrase in all_phrases_unique:
        word_count = len(phrase.split())
        groups[word_count].append(phrase)

    phrases_to_observe = []
    for length, phrases_in_group in groups.items():
        if len(phrases_in_group) > 10:
            sample = random.sample(phrases_in_group, 10)
            for phrase in sample:
                phrases_to_observe.append(phrase)
        else:
            for phrase in phrases_in_group:
                phrases_to_observe.append(phrase)
    print(phrases_to_observe)


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
        extraction_tool.extract_phrase_sentences(phrases_to_observe)
    extraction_tool.aggregate_dependency_data()


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
