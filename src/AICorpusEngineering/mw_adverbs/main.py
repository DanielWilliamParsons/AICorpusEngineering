from AICorpusEngineering.mw_adverbs.extract import ExtractAdverbs
from pathlib import Path
import argparse

def repo_root() -> Path:
    """Return the repository root."""
    # main.py is at src/AICorpusEngineering/mw_adverbs/
    return Path(__file__).resolve().parents[4]

def train():
    """
    Learn the parameters of a logistic regression classifier for
    extracted multiword adverbs.
    """
    parser = argparse.ArgumentParser(description="Train a logistic classifier for multiword adverbs")
    parser.add_argument("corpus_dir", type=Path, help="Input the root directory of your corpus folder.")
    args = parser.parse_args()

    filepath = args.corpus_dir.expanduser().resolve() / "ICNALE_W_CHN_A2_0_N100/W_CHN_PTJ0_038_A2_0.txt"

    extraction_tool = ExtractAdverbs(filepath)
    extraction_tool.load_conll_file()
    extraction_tool.run_extraction()
