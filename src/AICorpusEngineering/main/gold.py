from AICorpusEngineering.text_proc.text_proc import TextProc
from pathlib import Path
import argparse
import os
import importlib.resources as resources

def main():
    parser = argparse.ArgumentParser(description="Run the tagging pipeline for adverbs")
    parser.add_argument("root_dir", type=Path, help="Input the root directory of your corpus text files.")
    parser.add_argument("results_dir", type=Path, help="Input the directory where you wish to place the results of extracting corpus data.")
    parser.add_argument("results_file", default="results.ndjson", help="Name of the file to save results, ndjson format")

    args = parser.parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()

    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    process = TextProc(root_dir, results_dir, args.results_file)
    process.collect_lang("_ADV")
    process.aggregate()