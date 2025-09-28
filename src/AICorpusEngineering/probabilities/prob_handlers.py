import math

class MCQProbHandler:
    def __init__(self, logprobs = None):
        """
        This class handles calculating probabilities of answers to multiple choice questions
        in the form of single letter answers, A, B, C ... J
        It also calculates the perplexity of Chain of Thought reasoning outputs for multiple choice question
        The logprobs looks like this:
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
        TODO: Create a "validate_logprobs" function to ensure we always get this shape
        at least from llama-3.1-8b-instruct
        """
        if logprobs is not None:
            self.set_logprobs(logprobs)
        else:
            self.logprobs = None
        self.final_answer_token_index = None

    def set_logprobs(self, logprobs):
        self.logprobs = logprobs
        self.final_answer_token_index = None # Reset to none if a new set of logprobs is entered to avoid keeping an old result, and the possibility of there being no final answer output in the new logprobs results.
        self._find_answer_index(self.logprobs["content"]) # Find the final answer of the multiple choice question from the content - could remain None if no answer found

    def _find_answer_index(self, content):
        """
        Finds the index of the single answer A, B, C etc.
        Currently allows up to 10 possible answers from a multiple choice question from A - J.
        The final_answer_token_index will remain as None if no answer is found
        """
        for i in range(len(content) - 1, -1, -1):
            if content[i]["token"].strip() in ("A", "B", "C", "D", "E", "F", "G", "H", "I", 'J'):
                self.final_answer_token_index = i
                break

    def return_final_answer_token_index(self):
        return self.final_answer_token_index
        

    def calculate_reasoning_perplexity(self):
        """
        Perplexity calculation extracts the token_logprobs which are the logprobs for the actually token
        that made up the answer
        """
        content = self.logprobs["content"]

        # Get the index of the final answer token
        # If it has already been calculated
        if self.final_answer_token_index is None:
            self._find_answer_index(content)
        
        # If it is none, handle this... record an error
        if self.final_answer_token_index is None:
            # TODO - handle this as an error and record it as such
            return "No answer found; unable to compute perplexity for this question"
        
        token_entries = self.logprobs["content"][:self.final_answer_token_index] # Exclude the final answer
        token_logprobs = [entry["logprob"] for entry in token_entries] #Get the logprobs

        if not token_logprobs:
            # TODO - handle this as an error and record it as such
            return float("inf")
        
        nll = -sum(token_logprobs) / len(token_logprobs) # Negative log likelihood (average)
        ppl = math.exp(nll) # perplexity score
        return ppl

    def calculate_prob_distribution(self, choice_selections):
        """
        Calculates the probability distribution of the answer choices inside the language model.
        This distribution represents the probability of selecting one of the answer given the prompt and the chain of thought reasoning.
        When indicating choice_selections array, be careful to indicate the correct tokenization for the model used, otherwise the found tokens might be incorrect
        choice_selections: An array of letter choices for the multiple choices questions, e.g., [" A", " B", " C", " D", " E"] up to " J"
        """
        content = self.logprobs["content"]

        # Get the index of the final answer token
        # If it has already been calculated
        if self.final_answer_token_index is None:
            self._find_answer_index(content)
        
        # If the final answer index is None, handle this case.
        if self.final_answer_token_index is None:
            # TODO
            # Handle this situation! What should I have the pipeline do when no probs are found?
            return {k: 0.0 for k in choice_selections}
        
        entry = content[self.final_answer_token_index]
        top_entries = entry.get("top_logprobs", []) # extract the top_logprobs for all tokens that could be this token
        top_dict = {e["token"]: e["logprob"] for e in top_entries} # Create a dictionary of the top_logprobs

        # Build unnormalized probs for A/B/C/D from top_logprobs (handle variants)
        raw_probs = {}
        for ch in choice_selections:
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
            answer_probs = {k: 0.0 for k in choice_selections}
        return answer_probs