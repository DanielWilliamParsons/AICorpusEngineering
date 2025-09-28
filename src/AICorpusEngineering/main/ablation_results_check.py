from pathlib import Path
import argparse
import pandas as pd
import json

def repo_root() -> Path:
    """Return the repository root."""
    # adverbs.py is at src/AICorpusEngineering/main/
    return Path(__file__).resolve().parents[4]

def load_ndjson(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    """
    Analyze and summarize the results of the ablation study
    of gold sentences
    """
    parser = argparse.ArgumentParser(description="Run the results summary for the ablation study")
    # ----------
    # Get the user paths and filenames
    # ----------
    parser.add_argument("gold_standard_dir", type=Path, help="Input the name of the directory where the gold standard data is stored")
    parser.add_argument("gold_standard_filename", help="Input the name of the file containing the tagged gold standard sentences.")
    parser.add_argument("ablation_results_dir", type=Path, help="Input the directory of the ablation study results.")
    parser.add_argument("ablation_results_file", help="Input the name of the file containing the ablation study results.")

    # ----------
    # Resolve the user paths and filenames
    # ----------
    args = parser.parse_args()
    gold_standard_path = args.gold_standard_dir / args.gold_standard_filename
    ablation_results_path = args.ablation_results_dir / args.ablation_results_file
    gold_standard_path = gold_standard_path.expanduser().resolve()
    ablation_results_path = ablation_results_path.expanduser().resolve()

    # ----------
    # Read files
    # ----------
    gold_standard = load_ndjson(gold_standard_path)
    ablation_results = load_ndjson(ablation_results_path)

    # ----------
    # Convert to dataframe
    # ----------
    df_gold = pd.DataFrame(gold_standard)
    df_ablation = pd.DataFrame(ablation_results)

    print(df_gold)
    print(df_ablation)

if __name__ == "__main__":
    main()