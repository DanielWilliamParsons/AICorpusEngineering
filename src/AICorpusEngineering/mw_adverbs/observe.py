import pandas as pd
import os
from pathlib import Path
class ObserveAdverbs:
    
    def __init__(self, filepath, results_path):
        self.filepath = filepath
        self.sentences = []
        self.results_path = results_path
        self.phrase_df = pd.DataFrame(columns=["phrase", "length", "sentence", "initial_XPOS", "final_XPOS", "initial_DEPREL", "final_DEPREL", "initial_DEPS", "final_DEPS"])

    def load_conll_file(self):
        """ Load up the CoNLL-U file indiciated by the self.filepath """
        current_sent = []
        print(self.filepath)
        with open(self.filepath, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    if current_sent:
                        self.sentences.append(current_sent)
                        current_sent = []
                    continue
                parts = line.split("\t")
                if len(parts) != 9:
                       continue # skip malformed lines
                id, form, lemma, upos, xpos, feats, head, deprel, deps = parts
                current_sent.append({
                    "id": id,
                    "form": form,
                    "line": line,
                    "lemma": lemma,
                    "upos": upos,
                    "xpos": xpos,
                    "feats": feats,
                    "head": head,
                    "deprel": deprel,
                    "deps": deps
                })
        if current_sent:
            self.sentences.append(current_sent)

    def extract_phrase_sentences(self, phrases):
        """Extract sentences containing the given phrase"""
        for sent in self.sentences:
            words = [tok["form"].lower() for tok in sent]
            for phrase in phrases:
                phrase_words = phrase.lower().split()
                with self.results_path.open("a", encoding="utf-8-sig") as f:
                    for i in range(len(words) - len(phrase_words) + 1):
                        if words[i:i+len(phrase_words)] == phrase_words:
                            matched_tokens = sent[i:i+len(phrase_words)]
                            match_info = {
                                "phrase": " ".join(phrase_words),
                                "length": len(phrase_words),
                                "sentence": " ".join(words),
                                "initial_XPOS": matched_tokens[0]["xpos"],
                                "final_XPOS": matched_tokens[-1]["xpos"],
                                "initial_DEPREL": matched_tokens[0]["deprel"],
                                "final_DEPREL": matched_tokens[-1]["deprel"],
                                "initial_DEPS": matched_tokens[0]["deps"].split(":")[-1],
                                "final_DEPS": matched_tokens[-1]["deps"].split(":")[-1]
                            }
                            self.phrase_df.loc[len(self.phrase_df)] = match_info
                            f.write(f"----{' '.join(phrase_words)}----\n")
                            for word in sent:
                                f.write(word["line"] + "\n")
                            f.write("\n")
        phrase_df_path = self.results_path.with_suffix(".csv")
        file_exists = phrase_df_path.exists()
        self.phrase_df.to_csv(
            phrase_df_path,
            mode="a",
            header = not file_exists,
            index = False,
            encoding = "utf-8-sig"
        )
        self.phrase_df = pd.DataFrame(columns=self.phrase_df.columns)

                # for i in range(len(words) - len(phrase_words) + 1):
                #     if words[i:i+len(phrase_words)] == phrase_words:
                #         with self.results_path.open("a", encoding="utf-8") as f:
                #             f.write(f"----{phrase_words}----\n")
                #             for word in sent:
                #                 f.write(word["line"] + "\n")
                #             f.write("\n")
    
    def aggregate_dependency_data(self):
        """
        Aggregate the dependency data so we can count how frequency the combination of 
        DEPREL and DEPS are in initial and final position
        """
        phrase_df_path = self.results_path.with_suffix(".csv")
        df = pd.read_csv(phrase_df_path)
        agg_df = (
            df.groupby(["length", "initial_DEPREL", "final_DEPREL", "initial_DEPS", "final_DEPS"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        agg_df_path = self.results_path.with_name(self.results_path.stem + "_agg.csv")
        agg_df.to_csv(
            agg_df_path,
            mode="w",
            encoding = "utf-8-sig"
        )
    
    def aggregate_rules(self):
        """
        Use the results path to aggregate all the data in the _agg.csv files
        """
        results_dir = Path(self.results_path).parent
        file_paths = list(results_dir.rglob("*_agg.csv"))
        dfs = []
        for file in file_paths:
            df = pd.read_csv(file, encoding="utf-8-sig")
            df["source_file"] = file.name
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        agg_df = (
            combined_df.groupby(["length", "initial_DEPREL", "final_DEPREL", "initial_DEPS", "final_DEPS"], as_index=False)
                .agg({
                    "count": "sum",
                    "source_file": lambda x: ", ".join(sorted(set(x)))
                })
                .sort_values("count", ascending=False)
        )
        agg_df.to_csv(self.results_path, index=False, encoding="utf-8-sig")
