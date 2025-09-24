from pathlib import Path
import argparse
import os
import importlib.resources as resources
from datetime import datetime

from AICorpusEngineering.llm_server.server_manager import ServerManager
from AICorpusEngineering.agents.ablation_adverbs import AdverbsAblationStudy
from AICorpusEngineering.pipelines.ablation_adverbs import AblationPipeline
from AICorpusEngineering.probabilities.prob_handlers import MCQProbHandler
from AICorpusEngineering.knowledge_base.knowledge_base import KnowledgeBase
from AICorpusEngineering.logger.logger import NDJSONLogger
from AICorpusEngineering.logger.logger_registry import set_logger


def repo_root() -> Path:
    """Return the repository root."""
    # adverbs.py is at src/AICorpusEngineering/main/
    return Path(__file__).resolve().parents[4]

def get_chat_template_path() -> Path:
    """Return the installed path to the adverbs.jinja template."""
    return resources.files("AICorpusEngineering.agent-templates").joinpath("ablation_adverbs.jinja")


def resolve_repo_path(path_str: str) -> Path:
    """Helper to resolve relative paths against the repo root."""
    return (repo_root() / path_str).resolve()


def main():
    parser = argparse.ArgumentParser(description="Run the tagging pipeline for adverbs")

    # ----------
    # Necessary user inputs
    # ----------
    parser.add_argument("input_dir", type=Path, help="Input the name of the directory where your texts are stored.")

    # ----------
    # Error and Data Logging
    # ----------
    parser.add_argument(
        "--error_logs",
        type=Path,
        default=None,
        help="Path to the error logs file (default: inside output_dir with timestamped name)"
    )

    parser.add_argument(
        "--data_logs",
        type=Path,
        default=None,
        help="Path to the data logs file (default: inside output_dir with timestamped name)"
    )

    # ----------
    # Server and Model
    # ----------
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

    # ----------
    # Resolve user paths
    # ----------
    input_dir = args.input_dir.expanduser().resolve() # expanduser deals with ~ and resolve deals with relative paths

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # ----------
    # Create the logger
    # ----------
    logger = NDJSONLogger(args.data_logs, args.error_logs, args.output_dir)
    set_logger(logger) # Register a global instance of the logger, now available anywhere.

    # ----------
    # Prepare the agent template
    # ----------
    chat_template = get_chat_template_path()
    if not chat_template.exists():
        raise FileNotFoundError(f"Chat template not found at {chat_template}")
    
    # ----------
    # Prepare objects for ablation study and start server
    # ----------
    server = ServerManager(args.server_bin, args.model, chat_template)
    prob_handler = MCQProbHandler()
    knowledge_base = KnowledgeBase()
    server.start()

    # ----------
    # Begin the ablation studies
    # ----------
    try:
        agents = AdverbsAblationStudy(args.server_url, prob_handler, knowledge_base)
        pipeline = AblationPipeline(agents, logger)
        pipeline.run(input_dir, output_dir)
    finally:
        server.stop()


if __name__ == "__main__":
    main()