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
        print(f"\nC: Focus Adverb")
        print(f"\nD: Linking Adverb")
        print(f"\nE: Discourse Adverb")

        choice = input("\nChoose A, B, C, D or E: ").strip()
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
            result["main_tag"] = "FOCUS"
            print("\nChoose from the FOCUS adverbs: ")
            result["sub_tag"] = focus()
        if choice == "D":
            result["main_tag"] = "LINKING"
            print("\nChoose from the linking adverbs:")
            result["sub_tag"] = linking()
        if choice == "E":
            result["sub_tag"] = "DISCOURSE"
            print("\nChoose from the discourse adverbs: ")
            result["sub_tag"] = discourse()

        results_data.append(result)
        print(result)
        with path_to_save_data.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii = False) + "\n")

def circumstance():
    print("\nA: TIME")
    print("\nB: PLACE")
    print("\nC: MANNER")
    print("\nD: DEGREE")
    print("\nE: QUANTITY_EXTENT")
    print("\nF: FREQUENCY")
    print("\nG: DURATION")
    choice = input("\n Choose A, B, C, D, E, F, G: ")
    if choice == "A":
        return "TIME"
    if choice == "B":
        return "PLACE"
    if choice == "C":
        return "MANNER"
    if choice == "D":
        return "DEGREE"
    if choice == "E":
        return "QUANTITY_EXTENT"
    if choice == "F":
        return "FREQUENCY"
    if choice == "G":
        return "DURATION"

def stance():
    print("\nA: EPISTEMIC")
    print("\nB: ATTITUDE")
    print("\nC: INFERENCE")
    print("\nD: STYLE")
    choice = input("\nChoose A, B, C or D: ")
    if choice == "A":
        return "EPISTEMIC"
    if choice == "B":
        return "ATTITUDE"
    if choice == "C":
        return "INFERENCE"
    if choice == "D":
        return "STYLE"
    
def focus():
    print("\nA: ADDITIVE")
    print("\nB: FOCUS_EXCLUSIVE")
    print("\nC: FOCUS_PARTICULAR")
    print("\nD: SCOPE")
    choice = input("\nChoose A, B, C or D: ")
    if choice == "A":
        return "ADDITIVE"
    if choice == "B":
        return "FOCUS_EXCLUSIVE"
    if choice == "C":
        return "FOCUS_PARTICULAR"
    if choice == "D":
        return "SCOPE"


def linking():
    print("\nA: RESULT")
    print("\nB: CONTRAST_CONCESSION")
    print("\nC: ADDITION")
    print("\nD: ENUMERATION")
    print("\nE: SUMMATION")
    print("\nF: TRANSITION")
    choice = input("\nChoose A, B, C, D, E or F: ")
    if choice == "A":
        return "RESULT"
    if choice == "B":
        return "CONTRAST_CONCESSION"
    if choice == "C":
        return "ADDITION"
    if choice == "D":
        return "ENUMERATION"
    if choice == "E":
        return "SUMMATION"
    if choice == "F":
        return "TRANSITION"
    
def discourse():
    print("\nA: DISCOURSE_ORGANIZER")
    print("\nB: INTERPERSONAL")
    print("\nC: TEXT_DEIXIS")
    choice = input("\nChoose A or B: ")
    if choice == "A":
        return "DISCOURSE_ORGANIZER"
    if choice == "B":
        return "INTERPERSONAL"
    if choice == "C":
        return "TEXT_DEIXIS"


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

    