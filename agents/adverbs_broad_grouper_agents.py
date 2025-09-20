import requests, json, re

class BroadGrouperAgents:
    """
    This class contains calls to three LLM agents.
    The first agent attempts to categorize the adverb according to one of four super categories of adverb:
    Circumstantial, Stance, Linking and Discursive.
    The categorization is validated by a second agent.
    If the second agent disagrees with the first, a third agent is called to make the final decision.
    """
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
        """
        Receives a sentence and one of the adverbs from the sentence.
        Passes the information into the broad-grouper-agent LLM for analysis.
        The LLM should return a JSON string, but if extra strings are returned these are removed
        and the JSON string is preserved and transformed into JSON.
        This is then passed back into to the pipeline.
        """
        print("Analyzing with broad-grouper-agent")
        prompt = json.dumps({"sentence": sentence, "adverb": adverb})
        
        # Send the data to the LMM
        data = self._send_request(prompt, "broad-grouper-agent", temperature=0.0, n_predict=64)

        # Get the data back from the LMM
        raw = data["choices"][0]["message"]["content"].strip()
        # Strip anything that is not inside a JSON style string
        match = re.search(r"\{.*\}", raw, re.S)
        parsed = json.loads(match.group(0)) if match else {}

        # Append the original sentence and the original adverb to the JSON result from the LLM
        # for record-keeping, then pass back
        parsed["sentence"] = sentence
        parsed["adverb"] = adverb
        return parsed

    def validate(self, category):
        print("This method will call the validation agent.")

    def correct(self, validated):
        print("This method will call the correcting agent.")