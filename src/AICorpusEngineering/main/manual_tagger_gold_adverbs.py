from pathlib import Path
import argparse
import os
import json
import importlib.resources as resources
from datetime import datetime

def repo_rooe() -> Path:
    """Return the repo root."""
    return Path(__file__).resolve.parents[4]

def manually_tag(path_to_data, path_to_save_data):
    """
    Load the sentences and read into memory
    """
    sentences_data = []
    with path_to_data.open("r", encoding="utf-8") as f:
        for line in f:
            sentences_data.append(json.loads(line))

    results_data = []
    with path_to_save_data.open("r", encoding="utf-8") as f:
        for line in f:
            results_data.append(json.loads(line))

    
    for sentence_data in sentences_data[len(results_data):]:
        print(f"\nHere is your sentence. Please tag the adverb: {sentence_data['adverb']}")
        print(f"\n{sentence_data['sentence']}")
        print(f"\nCHOICES:")
        print(f"\nA: Circumstance Adverb")
        print(f"\nB: Stance Adverb")
        print(f"\nC: Linking Adverb")
        print(f"\nD: Discourse Adverb")

        choice = input("\nChoose A, B, C or D: ").strip()
        result = sentence_data
        if choice == "A":
            result["main_tag"] = "CIRCUMSTANCE"
            print("\nChoose from the circumstance adverbs:")
            result["sub_tag"] = circumstance()
        if choice == "B":
            result["main_tag"] = "STANCE"
            print("\nChoose from the stance adverbs:")
            result["sub_tag"] = stance()
        if choice == "C":
            result["main_tag"] = "LINKING"
            print("\nChoose from the linking adverbs:")
            result["sub_tag"] = linking()
        if choice == "D":
            result["sub_tag"] = "DISCOURSE"

        results_data.append(result)
        print(result)
        with path_to_save_data.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii = False) + "\n")

def circumstance():
    print("\nA: TIME")
    print("\nB: PLACE")
    print("\nC: MANNER")
    print("\nD: DEGREE")
    print("\nE: FREQUENCY")
    print("\nF: DURATION")
    choice = input("\n Choose A, B, C, D, E, F: ")
    if choice == "A":
        return "TIME"
    if choice == "B":
        return "PLACE"
    if choice == "C":
        return "MANNER"
    if choice == "D":
        return "DEGREE"
    if choice == "E":
        return "FREQUENCY"
    if choice == "F":
        return "DURATION"

def stance():
    print("\nA: EPISTEMIC")
    print("\nB: ATTITUDE")
    print("\nC: STYLE")
    choice = input("\nChoose A, B, or C: ")
    if choice == "A":
        return "EPISTEMIC"
    if choice == "B":
        return "ATTITUDE"
    if choice == "C":
        return "STYLE"


def linking():
    print("\nA: RESULT")
    print("\nB: CONTRAST_CONCESSION")
    print("\nC: ADDITION")
    print("\nD: ORGANIZATION")
    print("\nE: TRANSITION")
    choice = input("\nChoose A, B, C, D, or E: ")
    if choice == "A":
        return "RESULT"
    if choice == "B":
        return "CONTRAST_CONCESSION"
    if choice == "C":
        return "ADDITION"
    if choice == "D":
        return "ORGANIZATION"
    if choice == "E":
        return "TRANSITION"


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
    path_to_save_data = args.input_dir / args.file_name
    path_to_save_data = path_to_save_data.with_name(path_to_save_data.stem + f"_tagged").with_suffix(".ndjson")
    path_to_save_data.touch(exist_ok = True) # Create the file to save data if it doesn't exist.

    if not path_to_data.exists():
        raise FileNotFoundError(f"Path to data not found: {path_to_data}")
    
    manually_tag(path_to_data, path_to_save_data)

    