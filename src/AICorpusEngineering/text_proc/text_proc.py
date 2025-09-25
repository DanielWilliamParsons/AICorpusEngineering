import os
import re
import random
import math
import json
import pandas as pd
from collections import defaultdict, Counter
from tag_map import TagMapper
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
        self.tag = tag # Set the tag to be available to other methods
        results_file_path = self.results_dir / self.results_file_name
        records = [] # Will be a list of dictionaries: {lang, file, line, sentence}
        tag_pattern = re.compile(r'(\b\w+)\s*{tag}\b', re.IGNORECASE)

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
            counts = Counter()
            files = defaultdict(set)
            with results_file_path.open("r", encoding = "utf-8") as f:
                for line in f:
                    records.append(line)
            
            # Count the number of times
            for rec in records:
                counts[rec[self.tag]] += 1
                files[rec[self.tag]].add(rec["file"])