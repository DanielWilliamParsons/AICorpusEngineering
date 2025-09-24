import requests, json, re
from pathlib import Path
from datetime import datetime
from AICorpusEngineering.probabilities.prob_handlers import MCQProbHandler
from AICorpusEngineering.knowledge_base.knowledge_base import KnowledgeBase
from AICorpusEngineering.error_handler.error_handler import error_handler

class AdverbsAblationStudy:
    """
    This class monitors and edits agents
    in the adverbs ablation study
    """
    def __init__(
            self,
            server_url,
            prob_handler: MCQProbHandler,
            knowledge_base: KnowledgeBase
        ):
            print("intialize the class")
            self.server_url = server_url
            self.knowledge_base_cache = None
            self.prob_handler = prob_handler
            self.knowledge_base = knowledge_base

    def _send_request(self, payload, agent_type, knowledge_base, sentence, adverb, temperature=0.001, n_predict=128):
        try:
            response = requests.post(
                f"{self.server_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                data = json.dumps({
                        "messages": [{"role": "user", "content": payload}],
                        "chat_template_kwargs": {
                            "agent_type": agent_type, 
                            "knowledge_base": knowledge_base,
                            "sentence": sentence,
                            "adverb": adverb
                        },
                        "n_predict": n_predict,
                        "predict": n_predict,
                        "top_p": 0.85,
                        "logprobs": 1000,
                        "echo": False,
                        "stop": ["<|user|>", "<|system|>"]
                }),
                timeout = 30,
            )

            if response.status.code != 200:
                raise RuntimeError(f"Server error: {response.status_code} with body: {response.text[:200]}")
            return response.json()
        except Exception as e:
             # Delegate all error handling to the error_handler
             return error_handler.handle(
                  e,
                  context = {
                       "server_url": self.server_url,
                       "agent_type": agent_type,
                       "sentence": sentence,
                       "adverb": adverb
                  }
             )
    