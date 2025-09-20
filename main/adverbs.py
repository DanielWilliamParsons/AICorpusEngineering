from pathlib import Path
import argparse
from llm_server.server_manager import ServerManager
from agents.adverbs_broad_grouper_agents import BroadGrouperAgents
from pipelines.broad_grouper_pipeline import BroadGrouperPipeline
from logger.logger import NDJSONLogger

def resolve_repo_path(path_str: str) -> Path:
    repo_root = Path(__file__).resolve().parent.parent.parent
    return (repo_root / path_str).resolve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_txt", type=Path, help="Input TXT file with POS-tagged sentences")
    parser.add_argument("output_txt", type=Path, help="Output TXT file with JSON results")
    parser.add_argument(
        "--input_texts_dir",
        type=resolve_repo_path,
        default="tagged_texts",
        help="Path to the folder with the input text files, defaults to tagged_texts"
    )
    parser.add_argument(
        "--output_texts_dir",
        type=resolve_repo_path,
        default="output_texts",
        help="Path to the folder with the output text files, defaults to output_texts"
    )
    parser.add_argument(
        "--server_bin", 
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "llama.cpp" / "build" / "bin" / "llama-server",
        help="Path to the llama-server binary"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        help="Path to the large language model"
    )
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "agent-templates" / "adverbs.jinja",
        help="Path to the chat template file"
    )
    parser.add_argument(
        "--server_url",
        default="http://127.0.0.1:8080",
        help="Server URL (default: http://127.0.0.1:8080)"
    )

    
    args = parser.parse_args()
    input_txt = (args.input_texts_dir / args.input_txt).resolve()
    output_txt = (args.output_texts_dir / args.output_txt).resolve()

    server = ServerManager(args.server_bin, args.model, args.chat_template)
    logger = NDJSONLogger(args.output_txt)
    server.start()
    try:
        agents = BroadGrouperAgents(args.server_url)
        pipeline = BroadGrouperPipeline(agents, logger)
        pipeline.run(input_txt, output_txt)
    finally:
        server.stop()