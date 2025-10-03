from nltk.corpus import wordnet as wn
import spacy
import random

class ExtractAdverbs:
    """
    Exhaustively extract multiword adverb candidates from a dependency parsed corpus.
    """

    def __init__(self, filepath):
        """
        Set up the extractor
        """
        self.filepath = filepath
        self.sentences = []
        self.nlp = spacy.load("en_core_web_lg")

    def _is_multiword_adverb(self, phrase):
        # lowercase, replace spaces with underscores
        wn_phrase = phrase.lower().replace(" ", "_")
        synsets = wn.synsets(wn_phrase, pos=wn.ADV)
        return len(synsets) > 0 # True or False
    
    def _get_embedding(self, phrase):
        """ Get embedding for multiword phrase (average of token vectors). """
        doc = self.nlp(phrase)
        if len(doc) == 0:
            return [0.0] * self.nlp.vocab.vectors_length
        return doc.vector

    def load_conll_file(self):
        current_sent = []
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
                    "lemma": lemma,
                    "upos": upos,
                    "xpos": xpos,
                    "feats": feats,
                    "head": head,
                    "deprel": deprel,
                    "deps": deps,
                })
        if current_sent:
            self.sentences.append(current_sent)

        print(self.sentences)

    def extract_features(self, sentence):
        """Extract candidate multiple word adverbs as spaCy spans and set up feature vector."""
        feature_rows = []

        for i, token in enumerate(sentence):
            # ADP start
            if token["upos"] == "ADP":
                span = [token]
                j = i + 1
                # collect DT/ADJ/NOUN until break
                while j < len(sentence) and sentence[j]["deprel"] in {"det", "amod", "pobj", "compound"}:
                    span.append(sentence[j])
                    if sentence[j]["deprel"] == "pobj":
                        # Check external relation of pobj
                        # split deps like "6:npadvmod|13:obl"
                        relations = []
                        if sentence[j]["deps"] and sentence[j]["deps"] != "_":
                            relations = [rel.split(":")[1] for rel in sentence[j]["deps"].split("|") if ":" in rel]

                        # --- PHRASE ----
                        phrase = " ".join(tok["form"] for tok in span)

                        # ---- LABEL ----
                        label = 1 if self._is_multiword_adverb(phrase) else 0

                        # ---- FEATURE VECTOR ----
                        features = {
                            # lexical
                            "phrase": phrase,
                            "first_word": span[0]["form"].lower(),
                            "last_word": span[-1]["form"].lower(),
                            "length": len(span),
                            "pos_pattern": "_".join(tok["upos"] for tok in span),
                            "dep_pattern": "_".join(tok["deprel"] for tok in span),
                            "external_relations": "|".join(relations) if relations else "none",
                            "embedding": self._get_embedding(phrase)
                        }

                        feature_rows.append((features, label))

                    j += 1
        return feature_rows
        
        
        
    def run_extraction(self):
        """
        Run the rule-based extraction
        """
        text_features = []
        for sent in self.sentences:
            text_features.extend(self.extract_features(sent))

        positives = [row for row in text_features if row[1] == 1]
        negatives = [row for row in text_features if row[1] == 0]

        if positives:
            if len(negatives) > len(positives):
                negatives = random.sample(negatives, len(positives))

            balanced_rows = positives + negatives
            random.shuffle(balanced_rows) # mix up the rows

        print(balanced_rows)
            