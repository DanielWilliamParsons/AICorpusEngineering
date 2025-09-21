import requests, json, re
from pathlib import Path
from datetime import datetime

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
        self.knowledge_base_cache = None
    
    def _send_request(self, payload, agent_type, knowledge_base, temperature=0.001, n_predict=128):
        response = requests.post(
            f"{self.server_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "messages": [{"role": "user", "content": payload}],
                "chat_template_kwargs": {"agent_type": agent_type, "knowledge_base": knowledge_base},
                "n_predict": n_predict,
                "temperature": temperature,
                "top_p": 0.85,
                "stop": ["<|user|>", "<|system|>"]
            })
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.text}")
        return response.json()
    
    def _retrieve_knowledge_base(self):
        print("Retrieve the knowledge base for the agent")
        # Knowledge base looks like this:
        # KNOWLEDGE ABOUT ADVERB CATEGORIES
        # - CIRCUMSTANCE ADVERBS provide information about...
        # - STANCE ADVERBS provide information about ...

        # If knowledge already exists in the knowledge_base_cache, send that back
        if self.knowledge_base_cache is not None:
            print("Using knowledge-base cache")
            return self.knowledge_base_cache

        knowledge_base_path = Path(__file__).resolve().parent.parent / "knowledge_base" / "adverbs.json"
        try:
            with knowledge_base_path.open("r", encoding="utf-8") as kb_file:
                self.knowledge_base_cache = json.load(kb_file)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Knowledge base not found at {knowledge_base_path} at runtime.") from exc
        knowledge_base = "KNOWLEDGE ABOUT ADVERB CATEGORIES:\n\n"

        for category_name, category_info in self.knowledge_base_cache["Adverbials"].items():
            description = category_info["description"]
            title = f"- {category_name.upper()}"
            if "adverbs" not in title.lower():
                title += " ADVERBS"
            knowledge_base += f"{title}: {description}\n"

        self.knowledge_base_cache = knowledge_base # Update the cache
        print(f"Knowledge base prepared: {knowledge_base}")
        return knowledge_base
    
    def _parse_raw_to_json(self, raw_llm_response):
        """
        When the LLM is supposed to return JSON
        use this method to post-process the llm's response and
        ensure valid json
        """
        # TODO: add a json validity checker
        # If the output is not json log it as such and then continue with processing
        # So add error logging capability
        match = re.search(r"\{.*\}", raw_llm_response, re.S)
        parsed = json.loads(match.group(0)) if match else {}
        return parsed
    
    def analyze_adverb(self, sentence: str, adverb: str):
        """
        Receives a sentence and one of the adverbs from the sentence.
        Passes the information into the broad-grouper-agent LLM for analysis.
        The LLM should return a JSON string, but if extra strings are returned these are removed
        and the JSON string is preserved and transformed into JSON.
        This is then passed back into to the pipeline.
        The parsed data that gets returned should look like this:
        {
            "category": "CIRCUMSTANCE", 
            "reason": "The adverb 'marvelously' is an adverb of degree because it makes the adjective wonderful stronger."},
            "sentence": "What a marvelously wonderful day we are having.",
            "adverb": "marvelously"
        }
        """
        print(f"\n########Analyzing '{adverb}' with broad-grouper-agent.########")
        prompt = json.dumps({"sentence": sentence, "adverb": adverb})

        knowledge_base = self._retrieve_knowledge_base()
        
        # Send the data to the LMM
        data = self._send_request(prompt, "broad-grouper-agent", knowledge_base = knowledge_base, temperature=0.0, n_predict=128)

        # Get the data back from the LMM
        raw = data["choices"][0]["message"]["content"].strip()
        # Strip anything that is not inside a JSON style string
        parsed = self._parse_raw_to_json(raw)

        # Append the original sentence and the original adverb to the JSON result from the LLM
        # for record-keeping, then pass back
        parsed["sentence"] = sentence
        parsed["adverb"] = adverb
        parsed["time"] = datetime.now().isoformat()
        print(f"\nAnalyzed {adverb}:\n{parsed}")
        return parsed

    def validate(self, analysis_result):
        print("\n\n###checking result with validator-agent###")
        prompt = {k: analysis_result[k] for k in ("sentence", "adverb", "category")}
        knowledge_base = self._retrieve_knowledge_base() # Get the knowledge base (likely cached)
        data = self._send_request(prompt, "validator-agent", knowledge_base = knowledge_base, temperature=0.0, n_predict=128)
        raw = data["choices"][0]["message"]["content"].strip()


    def correct(self, validated):
        print("This method will call the correcting agent.")
