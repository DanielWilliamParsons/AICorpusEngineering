
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
                if len(parts) != 5:
                       continue # skip malformed lines
                form, upos, xpos, head, dep = parts
                current_sent.append({
                    "form": form,
                    "upos": upos,
                    "xpos": xpos,
                    "head": head,
                    "dep": dep
                })
        if current_sent:
            self.sentences.append(current_sent)

        print(self.sentences)

    def extract_candidates(self, sentence):
        """Extract candidate mulword adverbs as spaCy spand."""
        candidates = []
        
        for i, token in enumerate(sentence):
            # Case 1: ADP starting a phrase:
            if token["upos"] == "ADP":
                # Collect subtree-like span: here, just take consecutive tokens until dependency breaks
                span = [token["form"]]
                j = i + 1
                while j < len(sentence) and sentence[j]["dep"] in {"det", "amod", "pobj", "compound"}:
                    span.append(sentence[j]["form"])
                    j += 1
                if len(span) > 1:
                    candidates.append(" ".join(span))
            if token["upos"] == "ADV" and token["dep"] == "pobj":
                head = token["head"]
                head_token = next((t for t in sentence if t["form"] == head), None)
                if head_token and head_token["upos"] == "ADP":
                    # get ADP + ADV chunk
                    span = [head_token["form"], token["form"]]
                    candidates.append(" ".join(span))

            print(candidates)
            return candidates
        
    def run_extraction(self):
        """
        Run the rule-based extraction
        """
        for sent in self.sentences:
            candidates = self.extract_candidates(sent)
            if candidates:
                print("Sentence:", " ". join(token["form"] for token in sent))
                print("Candidates: ", candidates)