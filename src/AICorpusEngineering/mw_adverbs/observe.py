
class ObserveAdverbs:
    
    def __init__(self, filepath, results_path):
        self.filepath = filepath
        self.sentences = []
        self.results_path = results_path

    def load_conll_file(self):
        """ Load up the CoNLL-U file indiciated by the self.filepath """
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
                    "line": line
                })
        if current_sent:
            self.sentences.append(current_sent)

    def extract_phrase_sentences(self, phrases):
        """Extract sentences containing the given phrase"""
        for sent in self.sentences:
            print(sent)
            words = [tok["form"].lower() for tok in sent]
            for phrase in phrases:
                phrase_words = phrase.lower().split()
                with self.results_path.open("a", encoding="utf-8-sig") as f:
                    for i in range(len(words) - len(phrase_words) + 1):
                        if words[i:i+len(phrase_words)] == phrase_words:
                            f.write(f"----{' '.join(phrase_words)}----\n")
                            for word in sent:
                                f.write(word["line"] + "\n")
                            f.write("\n")

                # for i in range(len(words) - len(phrase_words) + 1):
                #     if words[i:i+len(phrase_words)] == phrase_words:
                #         with self.results_path.open("a", encoding="utf-8") as f:
                #             f.write(f"----{phrase_words}----\n")
                #             for word in sent:
                #                 f.write(word["line"] + "\n")
                #             f.write("\n")
            