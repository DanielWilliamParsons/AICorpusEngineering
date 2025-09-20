from pathlib import Path
import argparse
from llm_server.server_manager import ServerManager
from agents.adverbs_broad_grouper_agents import BroadGrouperAgents
from pipelines.broad_grouper_pipeline import BroadGrouperPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_txt", type=Path, help="Input TXT file with POS-tagged sentences")
    parser.add_argument("output_txt", type=Path, help="Output TXT file with JSON results")
    parser.add_argument(
        "--server_bin", 
        type=Path,
        default=Path(__file__).resolve().parent.parent / "llama.cpp" / "build" / "bin" / "llama-server",
        help="Path to the llama-server binary"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        help="Path to the large language model"
    )
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=Path(__file__).resolve().parent / "agent-templates" / "adverbs.jinja",
        help="Path to the chat template file"
    )
    parser.add_argument(
        "--server_url",
        default="http://127.0.0.1:8080",
        help="Server URL (default: http://127.0.0.1:8080)"
    )

    args = parser.parse_args()

    server = ServerManager(args.server_bin, args.model, args.chat_template)
    server.start()
    try:
        agents = BroadGrouperAgents(args.server_url)
        pipeline = BroadGrouperPipeline(agents)
        pipeline.run(args.input_txt, args.output_txt)
    finally:
        server.stop()