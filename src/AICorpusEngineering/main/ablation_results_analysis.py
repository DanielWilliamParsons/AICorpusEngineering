from pathlib import Path
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np


def repo_root() -> Path:
    """Return the repository root."""
    # adverbs.py is at src/AICorpusEngineering/main/
    return Path(__file__).resolve().parents[4]

def normalize_labels(series):
    return series.str.replace(r"\s+ADVERBS$", "", regex=True)

def analyze_metrics(df, study_cols):
    """Compute accuracy, precision, recall, and F1 per study."""
    results = []
    y_true = df["main_tag"].dropna()  # gold labels
    for col in study_cols:
        raw_pred = df.loc[y_true.index, f"{col}_category"]
        y_pred = normalize_labels(raw_pred)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        results.append({
            "study": col,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    return pd.DataFrame(results)

def plot_confusion_matrix(df, study_col, labels, outpath=None):
    """
    Plot confusion matrix for one study vs. gold labels.
    """
    y_true = df["main_tag"].dropna()
    raw_pred = df.loc[y_true.index, f"{study_col}_category"]
    y_pred = normalize_labels(raw_pred)

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Gold label")
    plt.title(f"Confusion Matrix: {study_col}")
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()
    else:
        plt.show()


def per_category_report(df, study_col, labels, outpath=None):
    y_true = df["main_tag"].dropna()
    raw_pred = df.loc[y_true.index, f"{study_col}_category"]
    y_pred = normalize_labels(raw_pred)

    print(f"\n=== Detailed report for {study_col} ===")
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    if outpath:
        report_df.to_csv(outpath)
    else:
        print(f"\n=== Detailed report for {study_col} ===")
        print(report_df)

def plot_confidence_vs_correctness_multi(df, study_cols, outpath=None, n_bins=10):
    """
    Plot calibration curves for multiple studies on the same plot.
    """
    bins = np.linspace(0, 1, n_bins+1)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for study_col in study_cols:
        conf_col = f"{study_col}_P(answer_given)"
        correct_col = f"{study_col}_correct"

        # Drop rows without probabilities
        data = df[[conf_col, correct_col]].dropna()

        if data.empty:
            continue

        # Bin confidences
        data["bin"] = pd.cut(data[conf_col], bins)

        # Compute accuracy per bin
        grouped = data.groupby("bin").agg(
            mean_confidence=(conf_col, "mean"),
            accuracy=(correct_col, "mean"),
            count=(correct_col, "count")
        ).reset_index()

        # Plot curve
        plt.plot(grouped["mean_confidence"], grouped["accuracy"], "o-", label=study_col)

    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Confidence vs Correctness across studies")
    plt.legend()
    plt.tight_layout()
    
    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()
    else:
        plt.show()


def add_correctness_flags(df, study_cols, gold_label_col="main_tag"):
    for col in study_cols:
        preds = normalize_labels(df[f"{col}_category"])
        gold = df[gold_label_col]
        df[f"{col}_correct"] = (preds==gold).astype(int)
    return df


def main():
    """
    Analyze and summarize the results of the ablation study
    of gold sentences
    """
    parser = argparse.ArgumentParser(description="Run the results analysis for the ablation study.")
    # ----------
    # Get the user paths and filenames
    # ----------
    parser.add_argument("ablation_results_dir", type=Path, help="Input the directory of the ablation study results.")

    # ----------
    # Resolve the user paths and filenames
    # ----------
    args = parser.parse_args()
    aggregate_results_path = args.ablation_results_dir / "aggregate_results.pkl"
    df_expanded = pd.read_pickle(aggregate_results_path)
    print(df_expanded)
    
    # Analysis output directory
    outdir = args.ablation_results_dir / "analysis_outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    # List of study columns (adjust if you renamed them after expansion)
    study_cols = ["base_study", "kb_oneshot_cot", "kb_zeroshot", "zeroshot", "oneshot_cot", "fewshot_cot"]
    metrics_df = analyze_metrics(df_expanded, study_cols)

    print("\n=== Accuracy / Precision / Recall / F1 per study ===")
    print(metrics_df)
    metrics_df.to_csv(outdir / "summary_metrics.csv", index=False)

    labels = ["CIRCUMSTANCE", "STANCE", "FOCUS", "LINKING", "DISCOURSE"]
    # Per study outputs
    for col in study_cols:
        # Confusion matrix
        cm_path = outdir / f"{col}_confusion_matrix.png"
        plot_confusion_matrix(df_expanded, col, labels, outpath=cm_path)

        # Per category report
        report_path = outdir / f"{col}_classification_report.csv"
        per_category_report(df_expanded, col, labels, outpath=report_path)

    # Confidence vs correctness plot
    df_with_correctness = add_correctness_flags(df_expanded, study_cols)
    plot_confidence_vs_correctness_multi(df_with_correctness, study_cols, outpath=outdir / "calibration_plot.png")