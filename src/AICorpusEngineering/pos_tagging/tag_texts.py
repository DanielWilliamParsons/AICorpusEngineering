import os
import sys
import spacy
import spacy_udpipe
import re
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor

class SpacyTagger:
    # Defaults to smallest rule based tagger, but can be upgraded to "en_core_web_trf" (transformer) or "en_core_wb_md"
    # Retrieve on the command line with python3 -m spacy download en_core_web_trf
    # For udpipe, on CLI: python -m spacy_udpipe download en for parsing model.
    def __init__(self, model: str = "en_core_web_sm", parse: bool = False, udpipe: bool = False):
        print(f"Creating spaCy tagger with model {model} | Dependency parse: {parse}")
        if udpipe:
            self.nlp = self.load_udpipe("en")
        elif parse:
            self.nlp = spacy.load(model, disable=["ner"])
        else:
            self.nlp = spacy.load(model, disable=["ner", "parser"])
            self.nlp.add_pipe("sentencizer") # To allow parsing sentence by sentence

    def load_udpipe(self, lang: str = "en"):
        """
        Load a spaCy-UDPipe model, downloading it once if missing.
        """
        # Default cache path used by spacy-udpipe
        cache_dir = os.path.expanduser("~/.cache/spacy-udpipe")
        model_dir = os.path.join(cache_dir, lang)

        # If not already downloaded, fetch it
        if not os.path.exists(model_dir):
            print(f"Downloading UDPipe model for '{lang}'...")
            spacy_udpipe.download(lang)

        # Load from cache
        return spacy_udpipe.load(lang)
    
    def normalize_paragraphs(self, text: str):
        """Split the text into paragraphs by line breaks, collapsing multiple breaks"""
        text = re.sub(r'\n\s*\n+', '\n', text.strip()) # Collapses multiple new lines
        return text.splitlines()
    
    def tag_paragraph(self, paragraph: str) -> str:
        """Run spaCy tagging on a single paragraph. POS tagging only - (legacy mode)"""
        doc = self.nlp(paragraph)
        return " ".join(f"{token.text}_{token.pos_}" for token in doc)
    
    def tag_text_sentence_per_line(self, text: str) -> str:
        """Process text paragraph by paragraph, sentence by sentence (legacy mode)"""
        paragraphs = self.normalize_paragraphs(text)
        tagged_paragraphs = []
        for p in paragraphs:
            doc = self.nlp(p) # tag once at the paragraph level
            tagged_sentences = [
                " ".join(f"{token.text}_{token.pos_}" for token in sent)
                for sent in doc.sents if sent.text.strip()
            ]
            tagged_paragraphs.append("\n".join(tagged_sentences))
        return "\n\n".join(tagged_paragraphs) # blank line between paragraphs
    
    def tag_text_conllu(self, text: str) -> str:
        """Dependency parse in CoNLL-like format."""
        paragraphs = self.normalize_paragraphs(text)
        tagged_paragraphs = []
        for i, p in enumerate(paragraphs, 1):
            doc = self.nlp(p)
            tagged_sentences = []
            for sent in doc.sents:
                sent_lines = []
                for j, token in enumerate(sent, 1):
                    head_id = 0 if token.head == token else token.head.i - sent.start + 1
                    feats = token.morph if token.morph else "_"
                    external = f"{token.head.head.i - sent.start + 1}:{token.head.dep_}"
                    sent_lines.append(f"{j}\t{token.text}\t{token.lemma_}\t{token.pos_}\t{token.tag_}\t{feats}\t{head_id}\t{token.dep_}\t{external}\t")
                tagged_sentences.append("\n".join(sent_lines))
            tagged_paragraphs.append(f"# newpar id={i}\n" + "\n\n".join(tagged_sentences))
        return "\n\n".join(tagged_paragraphs)
    
    def tag_text(self, text: str) -> str:
        """Process full text by paragraphs"""
        paragraphs = self.normalize_paragraphs(text)
        tagged_paragraphs = [self.tag_paragraph(p) for p in paragraphs if p.strip()]
        return "\n".join(tagged_paragraphs)
    
class FileProcessor:
    def __init__(self, input_root: Path, output_root: Path, tagger: SpacyTagger = None):
        self.input_root = input_root
        self.output_root = output_root
        self.tagger = tagger

    def process_file(self, file_path: Path, parse: bool = False, udpipe: bool = False):
        """Read, tag, and save a single file."""
        print(f"Worker processing: {file_path}")
        if self.tagger is None:
            raise ValueError("No tagger available for processing files.")

        relative_path = file_path.relative_to(self.input_root)
        output_path = self.output_root / relative_path

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read, process, write
        with file_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            text = f.read()
        print(f"Tagging text in {f}")
        if udpipe:
            tagged_text = self.tagger.tag_text_conllu(text)
        elif parse:
            tagged_text = self.tagger.tag_text_conllu(text)
        else:
            tagged_text = self.tagger.tag_text_sentence_per_line(text)
        
        with output_path.open("w", encoding="utf-8") as f:
            f.write(tagged_text)

    def collect_files(self):
        """Recursively collect all files from input_root"""
        return[p for p in self.input_root.rglob("*") if p.is_file()]
    

def process_file_wrapper(args):
    """Wrapper to allow parallel execution with ProcessPoolExecutor"""
    file_path, input_root, output_root, model, parse, udpipe = args
    tagger = SpacyTagger(model=model, parse = parse, udpipe = udpipe)
    processor = FileProcessor(input_root, output_root, tagger)
    processor.process_file(file_path, parse = parse, udpipe = udpipe)

def main():
    parser = argparse.ArgumentParser(description="Tag text files using spaCy POS tagger")
    parser.add_argument("input_folder", type=Path, help="Input folder containing text files")
    parser.add_argument("output_folder", type=Path, help="Output folder to save tagged files")
    parser.add_argument(
        "--model",
        type=str,
        default="en_core_web_trf",
        help="spaCy model to use (e.g. en_core_web_sm, en_core_web_md, en_core_web_trf)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)"
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Enable full dependency parsing (default: POS only)"
    )
    parser.add_argument(
        "--udpipe",
        action="store_true",
        help="Enable depemdency parsing with Universal Dependencies"
    )
    args = parser.parse_args()

    input_root = args.input_folder.resolve()
    output_root = args.output_folder.resolve()

    if not input_root.exists() or not input_root.is_dir():
        print(f"Error: Input folder {input_root} does not exist or is not a directory.")
        sys.exit(1)

    # Collect files
    processor = FileProcessor(input_root, output_root)
    files = processor.collect_files()

    # Parallel execution with user-chosen model
    tasks = [(f, input_root, output_root, args.model, args.parse, args.udpipe) for f in files]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(process_file_wrapper, tasks))

    print(f"Processing complete. Tagged files saved to {output_root}")

if __name__ == "__main__":
    main()
