import json
from AICorpusEngineering.logger.logger import NDJSONLogger
from AICorpusEngineering.error_handler.error_handler import error_handler
from AICorpusEngineering.logger.logger_registry import get_logger

class AblationPipeline:
    """
    This class controls the classes and data flow for
    the adverbs ablation study
    """
    def __init__(self, ablation_agents_interface, logger: NDJSONLogger):
        self.ablation_agents_interface = ablation_agents_interface
        self.logger = get_logger() # Get the global instance of the logger
    
    def run(self, input_dir, output_dir):

        # ----------
        # Load gold standard sentences and append to an array
        # ----------
        sentences_data = []
        with input_dir.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sentences_data.append(json.loads(line))
        
        # ----------
        # Loop through each sentence and send to LLM for ablation study
        # ----------
        for i, line in enumerate(sentences_data, start = 1):
            sentence = line["sentence"]
            words = sentence.split()
            plain_sentence = " ".join(w.rsplit("_", 1)[0] if "_" in w else w for w in words)
            adverb = line["adverb"]
            try:
                result_by_syntax = self.ablation_agents_interface.base_study(plain_sentence, adverb)
                print(result_by_syntax)
                study_1_output = self.ablation_agents_interface.kb_oneshot_cot(plain_sentence, adverb)
                print(study_1_output)
                study_2_output = self.ablation_agents_interface.kb_zeroshot(plain_sentence, adverb)
                print(study_2_output)
                study_3_output = self.ablation_agents_interface.zeroshot(plain_sentence, adverb)
                print(study_3_output)
                study_4_output = self.ablation_agents_interface.oneshot_cot(plain_sentence, adverb)
                print(study_4_output)
                study_5_output = self.ablation_agents_interface.fewshot_cot(plain_sentence, adverb)
                print(study_5_output)
            except Exception as e:
                if error_handler:
                    error_handler.handle(e, context={"line": i, "sentence": plain_sentence, "adverb": adverb}) # Logging of the error is handled by the error_handler so no need to log
            
            # ----------
            # Check the final answer with the gold standard and record all the data
            # ----------
