[
  {
    "type": "ADP + NOUN",
    "examples": ["in fact", "in reality", "in style", "in vain", "in practice", "at home", "at sea", "at random"],
    "POS pattern": "ADP + (ADJ) + NOUN",
    "Dependency pattern": "Head = ADP, NOUN = pobj of ADP",
    "Whole phrase": "advmod or npadvmod of main predicate"
  },
  {
    "type": "ADP + DET + NOUN",
    "examples": ["in the long run", "in the meantime", "in the distance", "on the whole", "on average"],
    "POS pattern": "ADP + DET + (ADJ) + NOUN",
    "Dependency pattern": "ADP governs NOUN (pobj), DET and ADJ modify NOUN",
    "Whole phrase": "advmod / npadvmod of main verb"
  },
  {
    "type": "ADP + NOUN + ADP + NOUN",
    "examples": ["time after time", "day after day", "year after year"],
    "POS pattern": "NOUN + ADP + NOUN",
    "Dependency pattern": "First NOUN = head, second NOUN = pobj of ADP",
    "Whole phrase": "npadvmod (temporal iterative adverb)"
  },
  {
    "type": "ADP + PRON",
    "examples": ["to my mind", "to my surprise", "to my regret", "to my relief", "to my knowledge"],
    "POS pattern": "ADP + PRON (possessive) + NOUN",
    "Dependency pattern": "PRON = poss of NOUN, NOUN = pobj of ADP",
    "Whole phrase": "advcl or sentence-level advmod (stance/evaluative)"
  },
  {
    "type": "ADV + ADP + NOUN",
    "examples": ["as a matter of fact", "as a rule", "as usual"],
    "POS pattern": "ADV/SCONJ + ADP + DET + NOUN",
    "Dependency pattern": "Head = NOUN, linked via pobj; phrase modifies sentence",
    "Whole phrase": "advmod (stance/typicality/stance adverb)"
  },
  {
    "type": "ADP + ADV",
    "examples": ["by far", "at once", "at most", "at least"],
    "POS pattern": "ADP + ADV (sometimes with quantifier)",
    "Dependency pattern": "ADV modifies head (quantifier or adjective)",
    "Whole phrase": "advmod of adjective or verb"
  },
  {
    "type": "ADP + NOUN (plural idioms)",
    "examples": ["in most cases", "in any case", "in any event"],
    "POS pattern": "ADP + (DET) + NOUN (plural)",
    "Dependency pattern": "NOUN = pobj, DET modifies NOUN",
    "Whole phrase": "advmod or discourse connective"
  },
  {
    "type": "Lexicalized idioms (fixed chunks)",
    "examples": ["by heart", "by hand", "by no means", "without doubt", "without fail"],
    "POS pattern": "ADP + NOUN (fixed) / ADP + NEG + NOUN",
    "Dependency pattern": "Head = ADP, NOUN = pobj, NEG = advmod",
    "Whole phrase": "sentence-level advmod (certainty, limitation)"
  },
  {
    "type": "Quantifier phrase adverbs",
    "examples": ["a lot", "a little", "a bit", "a great deal more", "no longer"],
    "POS pattern": "DET + NOUN / DET + ADJ + NOUN",
    "Dependency pattern": "NOUN phrase functions adverbially",
    "Whole phrase": "advmod of verb"
  },
  {
    "type": "Conjunctive adverbials (linkers)",
    "examples": ["on the other hand", "in contrast", "by comparison", "in conclusion", "in short", "in summary"],
    "POS pattern": "ADP + DET + NOUN (multiword with linking semantics)",
    "Dependency pattern": "NOUN = pobj, full PP functions as discourse connective",
    "Whole phrase": "discourse-level advmod"
  },
  {
    "type": "Sequence markers",
    "examples": ["first of all", "second of all", "in the first place", "in the second place", "last but not least"],
    "POS pattern": "ORDINAL + ADP + DET + NOUN",
    "Dependency pattern": "NOUN = pobj, ordinal modifies",
    "Whole phrase": "discourse advmod (sequencing)"
  },
  {
    "type": "Negative polarity adverbs",
    "examples": ["not at all", "not in the least", "by no means", "in no way"],
    "POS pattern": "NEG + ADP + DET + NOUN / ADP + NEG + NOUN",
    "Dependency pattern": "NEG modifies phrase, NOUN = pobj",
    "Whole phrase": "sentence-level advmod (stance/degree negation)"
  }
]
