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

answer_map = {
    "A": "CIRCUMSTANCE",
    "B": "STANCE",
    "C": "FOCUS",
    "D": "LINKING",
    "E": "DISCOURSE"
}

def expand_study(df, study_col, gold_label_col="main_tag"):
    study_name = study_col # Keep the same name

    # Extract components into new columns
    df[f"{study_name}_final_answer"] = df[study_col].apply(
        lambda x: x.get("final_answer") if isinstance(x, dict) else None
    )
    df[f"{study_name}_category"] = df[study_col].apply(
        lambda x: x.get("category") if isinstance(x, dict) else None
    )
    df[f"{study_name}_CoT"] = df[study_col].apply(
        lambda x: x.get("CoT") if isinstance(x, dict) else None
    )
    df[f"{study_name}_probdist"] = df[study_col].apply(
        lambda x: x.get("probdist") if isinstance(x, dict) else None
    )

    # probability of answer given
    df[f"{study_name}_P(answer_given)"] = df.apply(
        lambda row: row[f"{study_name}_probdist"].get(row[f"{study_name}_final_answer"], None)
        if isinstance(row[f"{study_name}_probdist"], dict) else None,
        axis=1
    )

    # Probability of correct answer (use mapping A-E category)
    df[f"{study_name}_P(correct)"] = df.apply(
        lambda row: row[f"{study_name}_probdist"].get(
            next((k for k, v in answer_map.items() if v == row[gold_label_col]), None),
            None
        ) if isinstance(row[f"{study_name}_probdist"], dict) and pd.notnull(row[gold_label_col]) else None,
        axis=1
    )

    return df

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

    # ----------
    # Resolve the user paths and filenames
    # ----------
    args = parser.parse_args()
    gold_standard_path = args.gold_standard_dir / args.gold_standard_filename
    gold_standard_path = gold_standard_path.expanduser().resolve()
    all_files = list(args.ablation_results_dir.glob("_data_*.ndjson"))
    

    # ----------
    # Read files
    # ----------
    gold_standard = load_ndjson(gold_standard_path)
    ablation_results = [] # Since there may be many results files as a result of stopping and starting the ablation studies midway.
    for file in all_files:
        data = load_ndjson(file)
        df = pd.DataFrame(data)
        ablation_results.append(df)
    ablation_results_all = pd.concat(ablation_results, ignore_index=True)

    # ----------
    # Convert to dataframe
    # ----------
    df_gold = pd.DataFrame(gold_standard)
    df_ablation = pd.DataFrame(ablation_results_all)

    print(df_gold)
    print(df_ablation)

    df_merged = pd.merge(
        df_gold,
        df_ablation,
        on="id",
        how="left"
    )

    study_cols = ["base_study", "kb_oneshot_cot", "kb_zeroshot", "zeroshot", "oneshot_cot", "fewshot_cot"]
    for col in study_cols:
        df_merged = expand_study(df_merged, col)
    
    df_expanded = df_merged.drop(columns=study_cols)

    print(df_expanded.head())
    
if __name__ == "__main__":
    main()