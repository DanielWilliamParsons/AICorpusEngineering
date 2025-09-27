import json
import os

class MWAdverbsPipeline:
    def __init__(self, MWAdverbsAgent):
        self.mw_adverbs_agent = MWAdverbsAgent

    def run(self, input_dir):
        for dirpath, _, filenames in os.walk(input_dir):
            for fname in filenames:
                if not fname.endswith(".txt"):
                    continue
                file_path = os.path.join(dirpath, fname)
                print(file_path)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        sentence = line.strip()
                        words = sentence.split()
                        plain_sentence = " ".join(w.rsplit("_", 1)[0] if "_" in w else w for w in words)
                        result = self.mw_adverbs_agent.get_mw_adverbs(plain_sentence)
                        print(result)
