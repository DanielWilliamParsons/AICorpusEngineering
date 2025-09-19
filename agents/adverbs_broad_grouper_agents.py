import requests, json, re

class BroadGrouperAgent:
    def __init__(self, server_url):
        self.server_url = server_url
    
    def _send_request(self, payload, agent_type, temperature=0.001, n_predict=128):
        response = requests.post(
            f"{self.server_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "messages": [{"role": "user", "content": payload}],
                "chat_template_kwargs": {"agent_type": agent_type},
                "n_predict": n_predict,
                "temperature": temperature,
                "top_p": 0.85,
                "stop": ["<|user|>", "<|system|>", "<|assistant|>"]
            })
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.text}")
        return response.json()
    
    def analyze_adverb(self, sentence, adverb):
        print("This method will call the agent that analyzes the adverb.")

    def validate(self, category):
        print("This method will call the validation agent.")

    def correct(self, validated):
        print("This method will call the correcting agent.")