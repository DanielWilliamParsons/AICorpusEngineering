import requests, json, re
from pathlib import Path
from datetime import datetime
from AICorpusEngineering.probabilities.prob_handlers import MCQProbHandler
from AICorpusEngineering.knowledge_base.knowledge_base import KnowledgeBase

class BroadGrouperAgent:
    """
    This class calls the syntactic-grouper large language model
    which classifies an adverb using an injected knowledge base, Chain of Thought reasoning about syntax
    and few shot examples about the classification.
    Classifications are:
    A. CIRCUMSTANCE
    B. STANCE
    C. LINKING
    D. DISCOURSE
    """
    def __init__(
            self, 
            server_url, 
            prob_handler: MCQProbHandler, 
            knowledge_base: KnowledgeBase
        ):
            self.server_url = server_url
            self.knowledge_base_cache = None
            self.prob_handler = prob_handler
            self.knowledge_base = knowledge_base
    
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
            "sentence": "the sentence passed in",
            "adverb": "the original adverb passed in",
            "category": "CIRCUMSTANCE, STANCE, LINKING, DISCOURSE - one of these categories determined by the LLM",
            "final_answer": "A, B, C, D - one of these associated with the category, given by the LLM",
            "CoT": "The chain of thought reasoning output carried out by the LLM",
            "ppl": float - perplexity of the chain of thought tokens, calculated by the MCQProbHandler object prob_handler
            "probdist": { "A": float, "B": float, "C": float, "D": float} - the normalized probability distribution of the answers selectable by the LLM when selecting the final answer
        }
        """
        print(f"\n########  GROUPING '{adverb}' with syntactic-grouper-agent.  ########")
        prompt = ""

        #knowledge_base = self._retrieve_knowledge_base()
        # Construct the knowledge base if we do not already have it in cache
        if self.knowledge_base_cache is None:
            self.knowledge_base.create_broad_adverb_knowledge_base()
            self.knowledge_base_cache = self.knowledge_base.get_knowledge_base()

        
        # Send the data to the LMM
        data = self._send_request(prompt, 
                                  "syntactic-grouper", 
                                  knowledge_base = self.knowledge_base_cache, 
                                  sentence = sentence, 
                                  adverb = adverb, 
                                  temperature=0.0,
                                  n_predict=128
                                )

        # Get the data back from the LMM
        raw = data["choices"][0]["message"]["content"].strip()
        logprobs = data["choices"][0]["logprobs"]

        ### Handle probabilities ###
        # Use prob_handlers to calculate reasoning complexity
        self.prob_handler.set_logprobs(logprobs)
        ppl = self.prob_handler.calculate_reasoning_perplexity()
        
        # Use prob_handlers class to calculate the probability distribution of the answer
        choice_selections = [" A", " B", " C", " D"] #Notice that these are written with a space to account for tokenization in the model (in this case llama)
        answer_probs = self.prob_handler.calculate_prob_distribution(choice_selections)

        ### Parse data for return ###
        parsed = {}
        # Add the sentence and adverb to the data to send back to the pipeline
        parsed["sentence"] = sentence
        parsed["adverb"] = adverb
        
        # Get the chain of thought from the LLM
        match = re.search(r"<\|assistant\|>(.*?)Final Answer", raw, re.DOTALL | re.IGNORECASE) # This assumes the output always starts with <|assistant|> and ends with Final Answer: 
        if match:
            parsed["CoT"] = match.group(1).strip()
        else:
            parsed["CoT"] = raw # If the output was different, just put the raw LLM output into the parsed object

        # Add the final answer token
        # First, get the final answer index token from the prob_handler because this class can find it
        final_answer_token_index = self.prob_handler.return_final_answer_token_index()
        parsed["final_answer"] =  logprobs["content"][final_answer_token_index]["token"].strip()

        # Add the category answer
        kb_mappings = self.knowledge_base.get_knowledge_base_mappings()
        parsed["category"] = self.knowledge_base.get_knowledge_base_mappings()[parsed["final_answer"]]

        # Add the perplexity to the output
        parsed["ppl"] = ppl

        # Add the answer probability distribution to the output
        parsed["probdist"] = answer_probs

        # Ad the time
        parsed["time"] = datetime.now().isoformat()
        print(f"\nAnalyzed {adverb}:\n{parsed}")
        return parsed
