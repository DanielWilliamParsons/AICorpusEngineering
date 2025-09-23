import requests, json, re
from pathlib import Path
from datetime import datetime
import math

class BroadGrouperAgent:
    """
    This class contains calls to three LLM agents.
    The first agent attempts to categorize the adverb according to one of four super categories of adverb:
    Circumstantial, Stance, Linking and Discursive.
    The categorization is validated by a second agent.
    If the second agent disagrees with the first, a third agent is called to make the final decision.
    """
    def __init__(self, server_url, prob_handlers):
        self.server_url = server_url
        self.knowledge_base_cache = None
        self.prob_handlers = prob_handlers # A ProbHandlers object
    
    def _send_request(self, payload, agent_type, knowledge_base, sentence, adverb, temperature=0.001, n_predict=128):
        response = requests.post(
            f"{self.server_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "messages": [{"role": "user", "content": payload}],
                "chat_template_kwargs": {"agent_type": agent_type, "knowledge_base": knowledge_base, "sentence": sentence, "adverb": adverb},
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
    
    def _retrieve_knowledge_base(self):
        print("Retrieve the knowledge base for the agent")
        # Knowledge base looks like this:
        # KNOWLEDGE ABOUT ADVERB CATEGORIES
        # A. CIRCUMSTANCE ADVERBS provide information about...
        # B. STANCE ADVERBS provide information about ...

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

        # Letter choices
        letter_choices = ["A", "B", "C", "D"]
        for idx, (category_name, category_info) in enumerate(self.knowledge_base_cache["Adverbials"].items()):
            description = category_info["description"]
            title = f"{letter_choices[idx]}. {category_name.upper()}"
            if "adverbs" not in category_name.lower():
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
        # Cleaned up leaked reserved token like <|reserved_special_token_213|>
        cleaned = re.sub(r"<\|.*?\|>", "", raw_llm_response)

        # Find the first { ... }
        match = re.search(r"\{.*?\}", cleaned, re.S)
        if not match:
            # TODO: handle this case in logging and method to where this is returned
            return {}
        try:
            parsed = json.loads(match.group(0)) if match else {}
            # TODO: add a json validity checker
            # If the output is not json log it as such and then continue with processing
            # So add error logging capability
            return parsed
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            # TODO Handle this case
            return {}
    
    def _parse_CoT_and_json(self, raw_llm_response):
        """
        When the LLM is supposed to return a chain-of-thought group of sentences
        followed by some JSON object, use this method to post-process the LLM's response and
        ensure valid json
        """
        print(raw_llm_response)
        # This pattern gets the CoT string followed by the JSON string
        # Assistant tags may or may not be present
        pattern = re.compile(
            r"(?:<\|assistant\|>)?\s*(.*?)\s*(\{.*\})\s*(?:</\|assistant\|>)?",
            re.DOTALL
        )

        match = pattern.search(raw_llm_response)
        if not match:
            #TODO
            # handle the case when no match is found
            return None # No match found
        
        chain_of_thought = match.group(1).strip()
        json_part = match.group(2).strip()
        data = json.loads(json_part)
        data["CoT"] = chain_of_thought
        return data
    
    def analyze_by_syntax(self, sentence: str, adverb: str):
        """
        Receives a sentence and one of the adverbs from the sentence.
        Passes the information into the broad-grouper-agent LLM for analysis.
        The LLM should return a JSON string, but if extra strings are returned these are removed
        and the JSON string is preserved and transformed into JSON.
        This is then passed back into to the pipeline.
        The parsed data that gets returned should look like this:
        {
            "category": "CIRCUMSTANCE", 
            "confidence": "0.85"
        }
        """
        print(f"\n########  GROUPING '{adverb}' with syntactic-grouper-agent.  ########")
        prompt = ""

        knowledge_base = self._retrieve_knowledge_base()
        
        # Send the data to the LMM
        data = self._send_request(prompt, "syntactic-grouper", knowledge_base = knowledge_base, sentence = sentence, adverb = adverb, temperature=0.0, n_predict=128)

        # Get the data back from the LMM
        raw = data["choices"][0]["message"]["content"].strip()
        logprobs = data["choices"][0]["logprobs"]

        # Use prob_handlers to calculate reasoning complexity
        self.prob_handlers.set_logprobs(logprobs)
        ppl = self.prob_handlers.calculate_reasoning_perplexity()
        
        # Use prob_handlers class to calculate the probability distribution of the answer
        choice_selections = [" A", " B", " C", " D"] #Notice that these are written with a space to account for tokenization in the model (in this case llama)
        answer_probs = self.prob_handlers.calculate_prob_distribution(choice_selections)

        print("Reasoning perplexity: ", ppl)
        print("Answer probability distribution: ", answer_probs)
        print(raw)
        # Strip anything that is not inside a JSON style string
        parsed = self._parse_raw_to_json(raw)

        # Append the original sentence and the original adverb to the JSON result from the LLM
        # for record-keeping, then pass back
        parsed["sentence"] = sentence
        parsed["adverb"] = adverb
        parsed["time"] = datetime.now().isoformat()
        print(f"\nAnalyzed {adverb}:\n{parsed}")
        return parsed
