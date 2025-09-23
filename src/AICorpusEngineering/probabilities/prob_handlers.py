import math

class ProbHandlers:
    def __init__(self, logprobs = None):
        """
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

    def set_logprobs(self, logprobs):
        self.logprobs = logprobs

    def _find_answer_index(self, content):
        """
        Finds the index of the single answer A, B, C etc.
        Currently allows up to 10 possible answers from a multiple choice question from A - J.
        """
        answer_idx = None
        for i in range(len(content) - 1, -1, -1):
            if content[i]["token"].strip() in ("A", "B", "C", "D", "E", "F", "G", "H", "I", 'J'):
                answer_idx = i
                break
        return answer_idx

    def calculate_reasoning_perplexity(self):
        """
        Perplexity calculation extracts the token_logprobs which are the logprobs for the actually token
        that made up the answer
        """
        content = self.logprobs["content"]
        answer_idx = self._find_answer_index(content)
        if answer_idx is None:
            # TODO - handle this as an error and record it as such
            return "No answer found; unable to compute perplexity for this question"
        
        token_entries = self.logprobs["content"][:answer_idx] # Exclude the final answer
        token_logprobs = [entry["logprob"] for entry in token_entries] #Get the logprobs

        if not token_logprobs:
            # TODO - handle this as an error and record it as such
            return float("inf")
        
        nll = -sum(token_logprobs) / len(token_logprobs) # Negative log likelihood (average)
        ppl = math.exp(nll) # perplexity score
        return ppl

    def calculate_prob_distribution(self):
        print("This will calculate the distribution of the probabilities")