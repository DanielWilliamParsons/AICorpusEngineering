from pathlib import Path
import argparse
import os
import importlib.resources as resources

from AICorpusEngineering.llm_server.server_manager import ServerManager
from AICorpusEngineering.agents.multiword_adverbs_tagger import MWAdverbs
from AICorpusEngineering.pipelines.mw_adverb_pipeline import MWAdverbsPipeline

def repo_root() -> Path:
    """Return the repository root."""
    # adverbs.py is at src/AICorpusEngineering/main/
    return Path(__file__).resolve().parents[4]

def get_chat_template_path() -> Path:
    """Return the installed path to the adverbs.jinja template."""
    return resources.files("AICorpusEngineering.agent-templates").joinpath("adverbs.jinja")


def main():
    parser = argparse.ArgumentParser(description="Run the LLM based multi-word adverb tagger.")
    parser.add_argument("input_dir", type=Path, help="Input the name of the root directory storing your POS tagged corpus")
    
    parser.add_argument(
        "--server_bin",
        type=Path,
        default=Path(
            os.environ.get("LLM_SERVER_BIN", repo_root() / "llama.cpp/build/bin/llama-server")
        ),
        help="Path to the llama-server binary (env: LLM_SERVER_BIN)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            os.environ.get("LLM_MODEL", repo_root() / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        ),
        help="Path to the large language model (env: LLM_MODEL)",
    )
    parser.add_argument(
        "--server_url",
        default="http://127.0.0.1:8080",
        help="Server URL (default: http://127.0.0.1:8080)",
    )

    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve() # expanduser deals with ~ and resolve deals with relative paths
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    chat_template = get_chat_template_path()

    if not chat_template.exists():
        raise FileNotFoundError(f"Chat template not found at {chat_template}")

    server = ServerManager(args.server_bin, args.model, chat_template)
    try:
        agents = MWAdverbs(args.server_url)
        pipeline = MWAdverbsPipeline(agents)
        pipeline.run(input_dir)
    finally:
        server.stop()

if __name__ == "__main__":
    main()