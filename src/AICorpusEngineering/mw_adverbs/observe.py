import pandas as pd
from pathlib import Path
import os
from nltk.corpus import wordnet as wn
class ObserveAdverbs:
    
    def __init__(self, filepath, results_path):
        self.filepath = filepath
        self.sentences = []
        self.results_path = results_path
        self.phrase_df = pd.DataFrame(columns=["phrase", "length", "sentence", "initial_XPOS", "final_XPOS", "initial_DEPREL", "final_DEPREL", "initial_DEPS", "final_DEPS", "first_word", "last_word", "in_list", "POS_before", "POS_after", "head_tok", "head_tok_upos", "head_tok_xpos", "punct_before", "punct_after", "wn_exists", "wn_head_pos", "wn_head_lexname", "wn_head_synset_count", "position_ratio", "dependency_string", "pos_string", "depth_to_root"])

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
        """
        Extract sentences containing the given phrase
        The user inserts a phrase such as "on the other hand"
        and this method extracts all sentences containing that phrase
        from a CoLLN-U formatted set of sentences.
        """
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
    
    def aggregate_dependency_data(self):
        """
        Aggregate the dependency data so we can count how frequent the combination of 
        DEPREL and DEPS are in initial and final position.
        
        Reads a csv file containing the phrase, sentence, XPOS, DEPREL and DEPS for initial and final words of the phrase all in one line of the csv
        There are many lines in the csv.
        The csv represents a particular initial-final pattern, e.g., ADP * ADV pattern where the first word of the phrase is ADP and the last word is ADV

        """
        phrase_df_path = self.results_path.with_suffix(".csv")
        df = pd.read_csv(phrase_df_path)
        agg_df = (
            df.groupby(["length", "initial_XPOS", "final_XPOS", "initial_DEPREL", "final_DEPREL", "initial_DEPS", "final_DEPS"])
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
            combined_df.groupby(["length", "initial_XPOS", "final_XPOS", "initial_DEPREL", "final_DEPREL", "initial_DEPS", "final_DEPS"], as_index=False)
                .agg({
                    "count": "sum",
                    "source_file": lambda x: ", ".join(sorted(set(x)))
                })
                .sort_values("count", ascending=False)
        )
        agg_df.to_csv(self.results_path, index=False, encoding="utf-8-sig")

    @staticmethod
    def get_wordnet_features(word):
        """
        Return WordNet lexical semantic info for a word.
        """
        synsets = wn.synsets(word)
        if not synsets:
            return {
                "wn_exists": False,
                "wn_pos": "",
                "wn_lexname": "",
                "wn_synset_count": 0
            }
        first_syn = synsets[0]
        wn_pos_map = {
            "n": "NOUN",
            "v": "VERB",
            "a": "ADJ",
            "s": "ADJ",
            "r": "ADV"
        }
        return {
            "wn_exists": True,
            "wn_pos": wn_pos_map.get(first_syn.pos(), first_syn.pos()),
            "wn_lexname": first_syn.lexname(),
            "wn_synset_count": len(synsets)
        }
    
    @staticmethod
    def dependency_depth(token, tokens_by_id):
        """
        Return number of edges from token to root
        """
        depth = 0
        current = token
        visited = set()
        while current["head"] != "0" and current["head"] not in visited:
            visited.add(current["id"])
            head_id = current["head"]
            if head_id not in tokens_by_id:
                break
            current = tokens_by_id[head_id]
            depth += 1
        return depth
    
    def extract_candidate_adverbs_with_rules(self, patterns_df, mw_adverbs):
        """
        Apply the rules inferred from the observations to extract
        candidate multiword adverbs

        Args:
            patterns_df (pd.DataFrame): columns are ["length", "initial_XPOS", "final_XPOS", "initial_DEPREL", "final_DEPREL", "initial_DEPS", "final_DEPS", "count", "source_file"]

        """

        for sent in self.sentences:
            tokens_by_id = {tok["id"]: tok for tok in sent}
            deprel = [tok["deprel"].lower() for tok in sent]
            deps = [tok["deps"].lower() for tok in sent]
            xpos = [tok["xpos"] for tok in sent]
            upos = [tok["upos"] for tok in sent]
            words = [tok["form"] for tok in sent]

            for row in patterns_df.itertuples(index=False):
                pattern_length= int(row.length)
                for i in range(len(deprel) - pattern_length + 1):
                    # --- Step 1: Surface level match ---
                    if deprel[i] == row.initial_DEPREL and deprel[i + pattern_length - 1] == row.final_DEPREL and deps[i].split(":")[-1] == row.initial_DEPS and deps[i + pattern_length - 1].split(":")[-1] == row.final_DEPS and xpos[i] == row.initial_XPOS and xpos[i + pattern_length - 1] == row.final_XPOS:
                        matched_tokens = sent[i: i+pattern_length]
                        phrase_token_ids = {tok["id"] for tok in matched_tokens}
                        match_info = {
                            "phrase": " ".join(words[i:i+pattern_length]),
                            "length": pattern_length,
                            "sentence": " ".join(words),
                            "initial_XPOS": matched_tokens[0]["xpos"],
                            "final_XPOS": matched_tokens[-1]["xpos"],
                            "initial_DEPREL": row.initial_DEPREL,
                            "final_DEPREL": row.final_DEPREL,
                            "initial_DEPS": row.initial_DEPS,
                            "final_DEPS": row.final_DEPS
                        }
                        # --- Step 2: Check what the phrase modifies:
                        # Keep phrases that modify verbs, adverbs and adjectives only (for now)
                        upos_allowed = {"VERB", "ADJ", "ADV"}

                        for tok in matched_tokens:
                            head_id = tok["head"]
                            #print(f"token={tok['form']} head_id={head_id}")
                            if head_id == '0':
                                continue # ROOT
                            if head_id in phrase_token_ids:
                                continue # internal link
                            head_tok = tokens_by_id.get(head_id)
                            #print("Head tok: ", head_tok)
                            if head_tok and head_tok["upos"] in upos_allowed:
                                # --- Step 3: Extract features
                                # --- 3.1: Lexical features
                                match_info["first_word"] = words[i]
                                match_info["last_word"] = words[i + pattern_length - 1]
                                match_info["in_list"] = 1 if " ".join(words[i:i+pattern_length]).lower() in mw_adverbs else 0
                                match_info["POS_before"] = xpos[i-1] if i - 1 >= 0 else ""
                                match_info["POS_after"] = xpos[i + pattern_length] if i + pattern_length < len(xpos) else ""
                                match_info["head_tok"] = head_tok["form"]
                                match_info["head_tok_upos"] = head_tok["upos"]
                                match_info["head_tok_xpos"] = head_tok["xpos"]
                                match_info["punct_before"] = 1 if i > 0 and upos[i-1] == "PUNCT" else 0
                                match_info["punct_after"] = 1 if i + pattern_length < len(upos) and upos[i + pattern_length] == "PUNCT" else 0

                                # --- 3.2: WordNet Lexical Semantics - decided to remove this
                                # wn_info = self.get_wordnet_features(head_tok["lemma"].lower())
                                # match_info["wn_head_exists"] = wn_info["wn_exists"]
                                # match_info["wn_head_pos"] = wn_info["wn_pos"]
                                # match_info["wn_head_lexname"] = wn_info["wn_lexname"]
                                # match_info["wn_head_synset_count"] = wn_info["wn_synset_count"]

                                # --- 3.3: Positional information
                                match_info["position_ratio"] = i / len(xpos)

                                # --- 3.4: Frequency data


                                # --- 3.5: Syntactic features
                                match_info["dependency_string"] = " ".join(deprel[i:i+pattern_length])
                                match_info["pos_string"] = " ".join(xpos[i:i+pattern_length])
                                depth_to_root = self.dependency_depth(tok, tokens_by_id)
                                match_info["depth_to_root"] = depth_to_root

                                self.phrase_df.loc[len(self.phrase_df)] = match_info
                                print(self.phrase_df)
                                break
        file_exists = os.path.isfile(self.results_path)
        self.phrase_df.to_csv(self.results_path, mode="a", header = not file_exists, index=False, encoding="utf-8-sig")

                            


                            
                            


