import os
import sys
import spacy
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

class SpacyTagger:
    # Defaults to smallest rule based tagger, but can be upgraded to "en_core_web_trf" (transformer) or "en_core_wb_md"
    # Retrieve on the command line with python3 -m spacy download en_core_web_trf
    def __init__(self, model: str = "en_core_web_sm"):
        print("Creating spacy Tagger!")
        self.nlp = spacy.load(model, disable=["ner", "parser"]) # Faster, POS only!

    def normalize_paragraphs(self, text: str):
        """Split the text into paragraphs by line breaks, collapsing multiple breaks"""
        text = re.sub(r'\n\s*\n+', '\n', text.strip()) # Collapses multiple new lines
        return text.splitlines()
    
    def tag_paragraph(self, paragraph: str) -> str:
        """Run spaCy tagging on a single paragraph."""
        doc = self.nlp(paragraph)
        return " ".join(f"{token.text}_{token.pos_}" for token in doc)
    
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

    def process_file(self, file_path: Path):
        """Read, tag, and save a single file."""
        print(f"Worker processing: {file_path}")
        if self.tagger is None:
            raise ValueError("No tagger available for processing files.")

        relative_path = file_path.relative_to(self.input_root)
        output_path = self.output_root / relative_path

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read, process, write
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            print(text)
        tagged_text = self.tagger.tag_text(text)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(tagged_text)

    def collect_files(self):
        """Recursively collect all files from input_root"""
        return[p for p in self.input_root.rglob("*") if p.is_file()]
    

def process_file_wrapper(args):
    """Wrapper to allow parallel execution with ProcessPoolExecutor"""
    file_path, input_root, output_root, model = args
    tagger = SpacyTagger(model=model)
    processor = FileProcessor(input_root, output_root, tagger)
    processor.process_file(file_path)

def main():
    if len(sys.argv) < 3:
        print("Usage: python tag_texts.py <input_folder> <output_folder>")
        sys.exit(1)

    input_root = Path(sys.argv[1]).resolve()
    output_root = Path(sys.argv[2]).resolve()

    if not input_root.exists() or not input_root.is_dir():
        print(f"Error: Input folder {input_root} does not exist or is not a directory.")
        sys.exit(1)
    
    # Collect files
    processor = FileProcessor(input_root, output_root)
    files = processor.collect_files()

    # Parallel execution
    tasks =[(f, input_root, output_root, "en_core_web_trf") for f in files] # Uses the transformers model, backed by RoBERTa-base, for highest accuracy, but slower that "en_core_wb_sm" and "en_core_wb_md" which uses word vectors
    # On my macbook air 24GB RAM, 4 workers spiked the RAM up to 19GB
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(process_file_wrapper, tasks))

    print(f"Processing complete. Tagged files saved to {output_root}")

if __name__ == "__main__":
    main()
