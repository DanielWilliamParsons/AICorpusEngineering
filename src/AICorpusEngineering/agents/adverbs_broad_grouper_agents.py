import requests, json, re
from pathlib import Path
from datetime import datetime
import math

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
        print(raw_llm_response)
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
    
    def _calculate_reasoning_perplexity(self, logprobs):
        """
        "logprobs": {
            "content": [
                {
                    "id": 27, 
                    "token": "1",
                    "bytes": [60],
                    "logprob": -3.2663880119798705e-05
                    "top_logprobs": [
                        {
                            "id": 27, 
                            "token": "1",
                            "bytes": [60],
                            "logprob": -3.2663880119798705e-05
                        },
                        {
                            "id": 524,
                            "token": "</",
                            "bytes": [60, 47]
                            "logprob": -11.929385768492
                        },
                        ...
                    ]
                }
            ]
        }
        Perplexity calculation extracts the token_logprobs which are the logprobs for the actually token
        that made up the answer
        """
        content = logprobs["content"]
        answer_idx = None
        for i in range(len(content) - 1, -1, -1):
            if content[i]["token"].strip() in ("A", "B", "C", "D"):
                answer_idx = i
                break
        token_entries = logprobs["content"][:answer_idx] # Exclude the final answer
        token_logprobs = [entry["logprob"] for entry in token_entries] #Get the logprobs

        if not token_logprobs:
            return float("inf")
        
        nll = -sum(token_logprobs) / len(token_logprobs) # Negative log likelihood (average)
        ppl = math.exp(nll) # perplexity score
        return ppl

    def _calculate_final_answer_probs(self, logprobs):
        """
        "logprobs": {
            "content": [
                {
                    "id": 27, 
                    "token": "1",
                    "bytes": [60],
                    "logprob": -3.2663880119798705e-05
                    "top_logprobs": [
                        {
                            "id": 27, 
                            "token": "1",
                            "bytes": [60],
                            "logprob": -3.2663880119798705e-05
                        },
                        {
                            "id": 524,
                            "token": "</",
                            "bytes": [60, 47]
                            "logprob": -11.929385768492
                        },
                        ...
                    ]
                }
            ]
        }
        Takes the last item in the content as this represents the final answer token.
        Looks at its "top_logprops" list to find scores for 'A', 'B', 'C', and 'D'
        Converts logits to probabilities with softmax and normalizes
        """
        # Find the final answer token:
        content = logprobs["content"]

        # Walk backwards through the logprobs array to find the decision step: last token that is exactly A/B/C/D ignoring whitespace
        answer_idx = None
        for i in range(len(content) - 1, -1, -1):
            if content[i]["token"].strip() in ("A", "B", "C", "D"):
                answer_idx = i
                break
        
        if answer_idx is None:
            # TODO
            # Handle this situation! What should I have the pipeline do when no probs are found?
            return {k: 0.0 for k in ["A", "B", "C", "D"]}
        
        entry = content[answer_idx]
        top_entries = entry.get("top_logprobs", []) # extract the top_logprobs for all tokens that could be this token
        top_dict = {e["token"]: e["logprob"] for e in top_entries} # Create a dictionary of the top_logprobs

        # Build unnormalized probs for A/B/C/D from top_logprobs (handle variants)
        raw_probs = {}
        for ch in [" A", " B", " C", " D"]:
            if ch in top_dict:
                raw_probs[ch] = math.exp(top_dict[ch])
            else:
                raw_probs[ch] = 0.0 # missing from top-k

        # Now normalize the probabilities
        normalization_constant = sum(raw_probs.values())
        if normalization_constant > 0:
            answer_probs = {k.strip(): v / normalization_constant for k, v in raw_probs.items()}
        else:
            # Fallback if none of the choices were found
            answer_probs = {k: 0.0 for k in ["A", "B", "C", "D"]}
        return answer_probs

    
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
        ppl = self._calculate_reasoning_perplexity(logprobs)
        answer_probs = self._calculate_final_answer_probs(logprobs)

        print("Reasoning perplexity: ", ppl)
        print("Answer probability distribution: ", answer_probs)
        # Strip anything that is not inside a JSON style string
        parsed = self._parse_raw_to_json(raw)

        # Append the original sentence and the original adverb to the JSON result from the LLM
        # for record-keeping, then pass back
        parsed["sentence"] = sentence
        parsed["adverb"] = adverb
        parsed["time"] = datetime.now().isoformat()
        print(f"\nAnalyzed {adverb}:\n{parsed}")
        return parsed

    def analyze_by_elimination(self, sentence: str, adverb: str):
        print("\n\n### Running DIAGNOSTIC-grouper agent###")
        knowledge_base = self._retrieve_knowledge_base()
        prompt = ""

        # Send the data to the LMM
        data = self._send_request(prompt, "diagnostic-grouper-agent", knowledge_base = knowledge_base, sentence = sentence, adverb = adverb, temperature = 0.0, n_predict=256)

        # Get the data back from the LMM
        raw = data["choices"][0]["message"]["content"].strip()
        logprobs = data["choices"][0]["logprobs"]
        ppl = self._calculate_reasoning_perplexity(logprobs)
        answer_probs = self._calculate_final_answer_probs(logprobs)

        print("Reasoning perplexity: ", ppl)
        print("Answer probability distribution: ", answer_probs)

        parsed = ""
        return parsed


    def mediate(self, original_result, validated_result):
        print("\n\n###MEDIATING the disagreement###")
        original_result["disagreement"] = validated_result["reason"]
        prompt = json.dumps(original_result)
        knowledge_base = self._retrieve_knowledge_base() # Get the knowledge base (likely cached)
        data = self._send_request(prompt, "mediator-agent", knowledge_base = knowledge_base, temperature=0.0, n_predict=256)
        raw = data["choices"][0]["message"]["content"].strip()
        print(f"##MEDIATOR OUTPUT: {raw}")
