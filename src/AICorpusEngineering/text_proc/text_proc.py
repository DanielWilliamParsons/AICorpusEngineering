import os
import re
import random
import math
import json
import pickle
import pandas as pd
from collections import defaultdict, Counter
from AICorpusEngineering.text_proc.tag_map import TagMapper
from pathlib import Path

class TextProc:
    """
    User must supplt a directory where the corpus text files are stored,
    a directory name relative to the current working directory to store the results,
    and a file name in which the store the results of the extraction.
    """

    def __init__(self, root_dir: Path, results_dir: Path, results_file_name: str):
        print("Initialize text processor")
        self.root_dir = root_dir.expanduser().resolve() # Root directory of the corpus of POS tagged texts
        self.tag_map = TagMapper()
        self.tag = None
        self.results_dir = results_dir.expanduser().resolve() # Location where text processing will be stored.
        self.results_dir.mkdir(parents=True, exist_ok=True) # Make sure the result directory exists
        self.results_file_name = results_file_name

    # ----------
    # Loop through corpus texts and extract language
    # ----------
    def collect_lang(self, tag: str):
        """
        Extracts sentences which contain language tagged with a specified tag value
        For example, specifying the tag as "_ADV" will extract sentences from the corpus which contain
        adverbs tagged with _ADV.
        Assumes a tagged text file contains one tagged sentence per line with line breaks for paragraphs
        TODO: Later introduce a flag to indicate this format, e.g., otspl = true / false
        """
        print("Beginning processing.")
        self.tag = tag # Set the tag to be available to other methods
        results_file_path = self.results_dir / self.results_file_name
        records = [] # Will be a list of dictionaries: {lang, file, line, sentence}
        tag_pattern = re.compile(rf'(\b\w+)\s*{tag}\b', re.IGNORECASE)

        # Can we parallelize this process?
        run_count = 0
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fname in filenames:
                file_path = os.path.join(dirpath, fname)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        sentence = line.strip()
                        for match in tag_pattern.finditer(sentence):
                            pos = match.group(1)
                            key = self.tag_map.map_tag(tag)
                            records.append({
                                key: pos.lower(),
                                "file": fname,
                                "line": line_num,
                                "sentence": sentence # Should I remove the POS tags?
                            })
                            run_count += 1
                            if run_count == 100:
                                with results_file_path.open("w", encoding="utf-8") as f:
                                    for record in records:
                                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                        print(record)
                                records = [] # reset the records array to empty
                                run_count = 0 # reset the run counter to 0
        # Store these records on disk for access later
        # because it is highly likely that there could be tens of thousands of them
        # and storing in memory would burden the computer
        if run_count > 0:
            # Save the remaining records to file
            with results_file_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")


    # ----------
    # Aggregrate the tags of interest
    # ----------
    def aggregate(self):
        results_file_path = self.results_dir / self.results_file_name
        if not results_file_path.exists():
            print(f"The file {results_file_path} does not exist. Try running collect_lang first where it will be created.")

        records = []
        if self.tag is not None:
            counts = defaultdict(int)
            files = defaultdict(set)
            with results_file_path.open("r", encoding = "utf-8") as f:
                for line in f:
                    records.append(json.loads(line))
            
            # Count the number of times
            for rec in records:
                key = self.tag_map.map_tag(self.tag)
                counts[rec[key]] += 1
                files[rec[key]].add(rec["file"])

            results_file_path_counts = results_file_path.with_name(results_file_path.stem + f"_{key}_counts").with_suffix(".pkl")
            results_file_path_files = results_file_path.with_name(results_file_path.stem + f"_{key}_files").with_suffix(".pkl")


            # Entropy-based spread: how evenly the part of speech is distributed across files
            spread_scores = {}
            for word, count in counts.items():
                file_dist = [sum(1 for r in records if r[key] == word and r["file"] == f)
                             for f in files[word]]
                total = sum(file_dist)
                probs = [c / total for c in file_dist]
                entropy = -sum(p * math.log(p + 1e-10) for p in probs) # Shannon Entropy
                spread_scores[word] = entropy
            print(spread_scores)

            df = pd.DataFrame({
                key: list(counts.keys()),
                "total_count": list(counts.values()),
                "file_count": [len(files[word]) for word in counts],
                "spread_score": [spread_scores[word] for word in counts]
            })
            results_file_path_df = results_file_path.with_name(results_file_path.stem + f"_{key}_df").with_suffix(".pkl")

            with open(results_file_path_df, "wb") as f:
                pickle.dump(df, f)
            with open(results_file_path_counts, "wb") as f:
                pickle.dump(df, f)
            with open(results_file_path_files, "wb") as f:
                pickle.dump(df, f)