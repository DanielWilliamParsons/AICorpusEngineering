from AICorpusEngineering.text_proc.text_proc import TextProc
from pathlib import Path
import argparse
import os
import importlib.resources as resources

def main():
    """
    Sample sentences from a corpus with the purpose of creating a gold standard.
    The sampling uses uniform random sampling of sentences that have been ranked according to
    the entropy value associated with the frequency of adverbs in the corpus relative to the
    frequency of texts containing the corpus.
    """
    parser = argparse.ArgumentParser(description="Run the tagging pipeline for adverbs")
    parser.add_argument("root_dir", type=Path, help="Input the root directory of your corpus text files.")
    parser.add_argument("results_dir", type=Path, help="Input the directory where you wish to place the results of extracting corpus data.")
    parser.add_argument("results_file", default="results.ndjson", help="Name of the file to save results, ndjson format")
    parser.add_argument(
        "--pos_tag",
        default = "_ADV",
        help = "Specify the pos tag for which you would like to sample sentences. Sentences containing the tag will be sampled. Default is _ADV"
    )
    parser.add_argument(
        "--sample_size",
        default=100,
        help = "Specify the number of sentences you wish to sample. Default is 100"
    )

    args = parser.parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()

    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    process = TextProc(root_dir, results_dir, args.results_file)
    process.collect_lang(args.pos_tag)
    process.aggregate()
    process.sample_words(n=args.sample_size)
    process.sample_sentences()