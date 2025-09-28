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
                        "messages": [],
                        "chat_template_kwargs": {
                            "agent_type": agent_type, 
                            "knowledge_base": knowledge_base,
                            "sentence": sentence,
                            "adverb": adverb
                        },
                        "n_predict": n_predict,
                        "temperature": temperature,
                        "top_p": 0.85,
                        "logprobs": 1000,
                        "echo": False,
                        "stop": ["<|user|>", "<|system|>"]
                }),
                timeout = 30,
            )

            if response.status_code != 200:
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
    
    def base_study(self, sentence: str, adverb: str):
        """
        Knowledge base + few-shot + CoT
        This is the baseline study.
        We use a knowledge base, and examples for each category
        which contain reasoning chains
        """
        print(f"\n------ Ablation Study: Beginning Base Study for {adverb} ------")

        # ----------
        # Prepare the prompt
        # ----------
        prompt = "" # No extra instructions in the study

        # ----------
        # Prepare the knowledge base
        # ----------
        if self.knowledge_base_cache is None:
            print("Creating knowledge base")
            self.knowledge_base.create_broad_adverb_knowledge_base()
            self.knowledge_base_cache = self.knowledge_base.get_knowledge_base()

        # ----------
        # Send the data to the LLM
        # ----------
        data = self._send_request(
            prompt,
            "base_study",
            knowledge_base = self.knowledge_base_cache,
            sentence = sentence,
            adverb = adverb,
            temperature = 0.0,
            n_predict = 256
        )
        if data is None:
            print(f"Request failed for adverb '{adverb}', see error logs")
            return None

        # ----------
        # Get the data back from the LLM and process the data
        # ----------
        raw = data["choices"][0]["message"]["content"].strip()
        print(raw)
        logprobs = data["choices"][0]["logprobs"]
        parsed = self.process_data(raw, logprobs, sentence, adverb, True) # Has chain of thought

        # ----------
        # Send processed data back to the pipeline
        # ----------
        print(f"\n------ Base Study for adverb '{adverb}': \n {parsed}")
        return parsed

    def kb_oneshot_cot(self, sentence: str, adverb: str):
        """
        Knowledge base + one-shot + CoT
        This is ablation study 1
        """
        print(f"\n------ Ablation Study: Beginning Ablation 1 - Knowledge base + one-shot + CoT for {adverb} ------")

        # ----------
        # Prepare the prompt
        # ----------
        prompt = "" # No extra instructions in the study

        # ----------
        # Prepare the knowledge base
        # ----------
        if self.knowledge_base_cache is None:
            self.knowledge_base.create_broad_adverb_knowledge_base()
            self.knowledge_base_cache = self.knowledge_base.get_knowledge_base()

        # ----------
        # Send the data to the LLM
        # ----------
        data = self._send_request(
            prompt,
            "kb_oneshot_cot",
            knowledge_base = self.knowledge_base_cache,
            sentence = sentence,
            adverb = adverb,
            temperature = 0.0,
            n_predict = 128
        )

        # ----------
        # Get the data back from the LLM and process the data
        # ----------
        raw = data["choices"][0]["message"]["content"].strip()
        logprobs = data["choices"][0]["logprobs"]
        parsed = self.process_data(raw, logprobs, sentence, adverb, True) # Has chain of thought

        # ----------
        # Send processed data back to the pipeline
        # ----------
        print(f"\n------ Ablation study 1 for adverb '{adverb}': \n {parsed}")
        return parsed
    
    def kb_zeroshot(self, sentence: str, adverb: str):
        """
        Knowledge base + zero shot
        This is ablation study 2
        In this study, we keep the knowledge base but use no examples
        """
        print(f"\n------ Ablation Study: Beginning Ablation 2 - Knowledge base + zero shot for {adverb} ------")

        # ----------
        # Prepare the prompt
        # ----------
        prompt = "" # No extra instructions in the study

        # ----------
        # Prepare the knowledge base
        # ----------
        if self.knowledge_base_cache is None:
            self.knowledge_base.create_broad_adverb_knowledge_base()
            self.knowledge_base_cache = self.knowledge_base.get_knowledge_base()

        # ----------
        # Send the data to the LLM
        # ----------
        data = self._send_request(
            prompt,
            "kb_zeroshot",
            knowledge_base = self.knowledge_base_cache,
            sentence = sentence,
            adverb = adverb,
            temperature = 0.0,
            n_predict = 128
        )

        # ----------
        # Get the data back from the LLM and process the data
        # ----------
        raw = data["choices"][0]["message"]["content"].strip()
        logprobs = data["choices"][0]["logprobs"]
        parsed = self.process_data(raw, logprobs, sentence, adverb, False) # Does not have chain of thought

        # ----------
        # Send processed data back to the pipeline
        # ----------
        print(f"\n------ Ablation study 2 for adverb '{adverb}': \n {parsed}")
        return parsed

    def zeroshot(self, sentence: str, adverb: str):
        """
        Zero shot only
        This is ablation study 3
        In this study, we offer no examples, just instructions
        """
        print(f"\n------ Ablation Study: Beginning Ablation 3 - Zero shot for {adverb} ------")

        # ----------
        # Prepare the prompt
        # ----------
        prompt = "" # No extra instructions in the study

        # ----------
        # Prepare the knowledge base 
        # In this study the knowledge base is just the category names
        # These are currently hard-coded into the jinja template
        # ----------
        # if self.knowledge_base_cache is None:
        #     self.knowledge_base.create_broad_adverb_knowledge_base()
        #     self.knowledge_base_cache = self.knowledge_base.get_knowledge_base()

        # ----------
        # Send the data to the LLM
        # ----------
        data = self._send_request(
            prompt,
            "zeroshot",
            knowledge_base = "",
            sentence = sentence,
            adverb = adverb,
            temperature = 0.0,
            n_predict = 128
        )

        # ----------
        # Get the data back from the LLM and process the data
        # ----------
        raw = data["choices"][0]["message"]["content"].strip()
        logprobs = data["choices"][0]["logprobs"]
        parsed = self.process_data(raw, logprobs, sentence, adverb, False) # Does not have chain of thought

        # ----------
        # Send processed data back to the pipeline
        # ----------
        print(f"\n------ Ablation study 3 for adverb '{adverb}': \n {parsed}")
        return parsed

    def oneshot_cot(self, sentence: str, adverb: str):
        """
        One shot only
        This is ablation study 4
        In this study, we offer one example with a reasoning
        chain and instructions
        """
        print(f"\n------ Ablation Study: Beginning Ablation 4 - One shot + CoT for {adverb} ------")

        # ----------
        # Prepare the prompt
        # ----------
        prompt = "" # No extra instructions in the study

        # ----------
        # Prepare the knowledge base 
        # In this study the knowledge base is just the category names
        # These are currently hard-coded into the jinja template
        # ----------
        # if self.knowledge_base_cache is None:
        #     self.knowledge_base.create_broad_adverb_knowledge_base()
        #     self.knowledge_base_cache = self.knowledge_base.get_knowledge_base()

        # ----------
        # Send the data to the LLM
        # ----------
        data = self._send_request(
            prompt,
            "oneshot_cot",
            knowledge_base = "",
            sentence = sentence,
            adverb = adverb,
            temperature = 0.0,
            n_predict = 128
        )

        # ----------
        # Get the data back from the LLM and process the data
        # ----------
        raw = data["choices"][0]["message"]["content"].strip()
        logprobs = data["choices"][0]["logprobs"]
        parsed = self.process_data(raw, logprobs, sentence, adverb, True) # Has chain of thought

        # ----------
        # Send processed data back to the pipeline
        # ----------
        print(f"\n------ Ablation study 4 for adverb '{adverb}': \n {parsed}")
        return parsed

    def fewshot_cot(self, sentence: str, adverb: str):
        """
        Few shot and chain of thought.
        This is ablation study 5.
        In this study, we offer examples for each category with
        reasoning chains and instructions, but no knowledge base.
        """
        print(f"\n------ Ablation Study: Beginning Ablation 5 - Few shot + CoT for {adverb} ------")

        # ----------
        # Prepare the prompt
        # ----------
        prompt = "" # No extra instructions in the study

        # ----------
        # Prepare the knowledge base 
        # In this study the knowledge base is just the category names
        # These are currently hard-coded into the jinja template
        # ----------
        # if self.knowledge_base_cache is None:
        #     self.knowledge_base.create_broad_adverb_knowledge_base()
        #     self.knowledge_base_cache = self.knowledge_base.get_knowledge_base()

        # ----------
        # Send the data to the LLM
        # ----------
        data = self._send_request(
            prompt,
            "fewshot_cot",
            knowledge_base = "",
            sentence = sentence,
            adverb = adverb,
            temperature = 0.0,
            n_predict = 128
        )

        # ----------
        # Get the data back from the LLM and process the data
        # ----------
        raw = data["choices"][0]["message"]["content"].strip()
        logprobs = data["choices"][0]["logprobs"]
        parsed = self.process_data(raw, logprobs, sentence, adverb, True) # Has chain of thought

        # ----------
        # Send processed data back to the pipeline
        # ----------
        print(f"\n------ Ablation study 5 for adverb '{adverb}': \n {parsed}")
        return parsed

    # ----------
    # Clear the knowledge base cache
    # This method can be called outside this class
    # so that the knowledge base cache can be cleared to allow a new knowledge base
    # to created if necessary
    # ----------
    def clear_knowledge_base_cache(self):
        """
        Necessary for reformulating the knowledge base for different studies
        """
        self.knowledge_base_cache = None

    # ----------
    # Process data retrieved back from the LLM
    # This utility method is called by all studies
    # ----------
    def process_data(self, raw_llm_output, logprobs, sentence: str, adverb: str, has_CoT: bool):
        """
        Data processing from the LLM is the same for each study
        """
        # ----------
        # Handle the probabilities
        # ----------
        self.prob_handler.set_logprobs(logprobs)
        ppl = self.prob_handler.calculate_reasoning_perplexity()
        choice_selections = [" A", " B", " C", " D", " E"]
        answer_probs = self.prob_handler.calculate_prob_distribution(choice_selections)

        # ----------
        # Parse data for returning to the pipeline
        # ----------
        parsed = {}
        parsed["sentence"] = sentence
        parsed["adverb"] = adverb

        if has_CoT:
            # Get the chain of thought from the LLM
            match = re.search(r"<\|assistant\|>(.*?)Final Answer", raw_llm_output, re.DOTALL | re.IGNORECASE) # This assumes the output always starts with <|assistant|> and ends with Final Answer: 
            if match:
                parsed["CoT"] = match.group(1).strip()
            else:
                parsed["CoT"] = raw_llm_output # If the output was different, just put the raw LLM output into the parsed object

        # Add the final answer token
        final_answer_token_index = self.prob_handler.return_final_answer_token_index()
        parsed["final_answer"] = logprobs["content"][final_answer_token_index]["token"].strip()

        # Add the category answer
        parsed["category"] = self.knowledge_base.get_knowledge_base_mappings()[parsed["final_answer"]]

        # Add the perplexity to the output
        parsed["ppl"] = ppl

        # Add the answer probability distribution to the output
        parsed["probdist"] = answer_probs

        # Add the time
        parsed["time"] = datetime.now().isoformat()
        return parsed
