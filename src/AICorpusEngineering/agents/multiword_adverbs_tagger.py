from pathlib import Path
from datetime import datetime
import requests, json

class MWAdverbs:
    """
    Thus class interfaces with the multiword adverb tagging large language model
    to look for multi-word adverbs in sentences.
    """

    def __init__(
            self,
            server_url,
    ):
        self.server_url = server_url

    def _send_request(self, payload, sentence, temperature=0.001, n_predict=128):
        response = requests.post(
            f"{self.server_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "messages": [{"role": "user", "content": payload}],
                "chat_template_kwargs": {"sentence": sentence},
                "n_predict": n_predict,
                "temperature": temperature,
                "top_p": 0.85,
                "logprobs": 1000,
                "echo": False,
                "stop": ["<|user|>", "<|system|>"]
            })
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.text}")
        return response.json()
    
    def get_mw_adverbs(self, sentence: str):
        """
        Receives a sentence from the user.
        Sends it to the LLM for processing.
        """
        print(f"\n---- ANALYZING sentence {sentence} ----")
        prompt = ""
        data = self._send_request(
            prompt,
            sentence = sentence,
            temperature = 0.0,
            n_predict = 256
        )

        raw = data["choices"][0]["message"]["content"].strip()
        return raw