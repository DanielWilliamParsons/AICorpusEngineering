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
    
    def base_study(self, sentence: str, adverb: str):
        """
        Knowledge base + few-shot + CoT
        This is the baseline study.
        We use a knowledge base, and examples for each category
        which contain reasoning chains
        """
        print(f"\n------ Ablation Study: Beginning Base Study ------")

        # ----------
        # Prepare the prompt
        # ----------
        prompt = ""

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
            "base_study",
            knowledge_base = self.knowledge_base_cache,
            sentence = sentence,
            adverb = adverb,
            temperature = 0.0,
            n_predict = 128
        )

        # ----------
        # Get the data back fromt he LLM
        # ----------
        raw = data["choices"][0]["message"]["content"].strip()
        logprobs =data["choices"][0]["logprobs"]

        # ----------
        # Handle the probabilities
        # ----------
        self.prob_handler.set_logprobs(logprobs)
        ppl = self.prob_handler.calculate_reasoning_perplexity()
        choice_selections = [" A", " B", " C", " D"]
        answer_probs = self.prob_handler.calculate_prob_distribution(choice_selections)

        # ----------
        # Parse data for returning to the pipeline
        # ----------
        parsed = {}
        parsed["sentence"] = sentence
        parsed["adverb"] = adverb

        # Get the chain of thought from the LLM
        match = re.search(r"<\|assistant\|>(.*?)Final Answer", raw, re.DOTALL | re.IGNORECASE) # This assumes the output always starts with <|assistant|> and ends with Final Answer: 
        if match:
            parsed["CoT"] = match.group(1).strip()
        else:
            parsed["CoT"] = raw # If the output was different, just put the raw LLM output into the parsed object

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

        print(f"\n------ Base Study for adverb '{adverb}': \n {parsed}")
        return parsed

    def kb_oneshot_cot(self, sentence: str, adverb: str):
        """
        Knowledge base + one-shot + CoT
        This is ablation study 1
        """
    
    def kb_zeroshot(self, sentence: str, adverb: str):
        """
        Knowledge base + zero shot
        This is ablation study 2
        In this study, we keep the knowledge base but use no examples
        """

    def zeroshot(self, sentence: str, adverb: str):
        """
        Zero shot only
        This is ablation study 3
        In this study, we offer no examples, just instructions
        """

    def oneshot_cot(self, sentence: str, adverb: str):
        """
        One shot only
        This is ablation study 4
        In this study, we offer one example with a reasoning
        chain and instructions
        """

    def fewshot_cot(self, sentence: str, adverb: str):
        """
        Few shot and chain of thought.
        This is ablation study 5.
        In this study, we offer examples for each category with
        reasoning chains and instructions, but no knowledge base.
        """

    def clear_knowledge_base_cache(self):
        """
        Necessary for reformulating the knowledge base for different studies
        """
        self.knowledge_base_cache = None