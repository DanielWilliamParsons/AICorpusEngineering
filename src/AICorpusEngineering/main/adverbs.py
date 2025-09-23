from pathlib import Path
import argparse
import os
import importlib.resources as resources

from AICorpusEngineering.llm_server.server_manager import ServerManager
from AICorpusEngineering.agents.adverbs_broad_grouper_agents import BroadGrouperAgents
from AICorpusEngineering.src.AICorpusEngineering.pipelines.tagging_pipeline import TaggingPipeline
from AICorpusEngineering.logger.logger import NDJSONLogger


def repo_root() -> Path:
    """Return the repository root."""
    # adverbs.py is at src/AICorpusEngineering/main/
    return Path(__file__).resolve().parents[4]

def get_chat_template_path() -> Path:
    """Return the installed path to the adverbs.jinja template."""
    return resources.files("AICorpusEngineering.agent-templates").joinpath("adverbs.jinja")


def resolve_repo_path(path_str: str) -> Path:
    """Helper to resolve relative paths against the repo root."""
    return (repo_root() / path_str).resolve()


def main():
    parser = argparse.ArgumentParser(description="Run the Adverbs Broad Grouper pipeline")
    parser.add_argument("input_txt", type=Path, help="Input TXT file with POS-tagged sentences")
    parser.add_argument("output_txt", type=Path, help="Output TXT file with JSON results")

    parser.add_argument(
        "--input_texts_dir",
        type=Path,
        default=resolve_repo_path("tagged_sample_texts"),
        help="Path to the folder with the input text files (default: Tagged Corpus Texts)",
    )
    parser.add_argument(
        "--output_texts_dir",
        type=Path,
        default=resolve_repo_path("output_texts"),
        help="Path to the folder for output text files (default: output_texts)",
    )
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

    input_txt = (args.input_texts_dir / args.input_txt).resolve()
    output_txt = (args.output_texts_dir / args.output_txt).resolve()
    chat_template = get_chat_template_path()

    if not chat_template.exists():
        raise FileNotFoundError(f"Chat template not found at {chat_template}")

    server = ServerManager(args.server_bin, args.model, chat_template)
    logger = NDJSONLogger(args.output_txt)
    server.start()
    try:
        agents = BroadGrouperAgents(args.server_url)
        pipeline = TaggingPipeline(agents, logger)
        pipeline.run(input_txt, output_txt)
    finally:
        server.stop()


if __name__ == "__main__":
    main()