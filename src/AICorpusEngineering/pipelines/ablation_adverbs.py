import json
from AICorpusEngineering.logger.logger import NDJSONLogger
from AICorpusEngineering.error_handler.error_handler import error_handler
from AICorpusEngineering.logger.logger_registry import get_logger
import time

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
        # Set up data logs and run completion logs
        # ----------
        completions = []
        logs_dir = self.logger.logs_dir # Get the directory to where the log files will be saved
        # Get the completion logs
        for run_completion in logs_dir.glob("_run_completion_*.ndjson"):
            with open(run_completion, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, start = 1):
                    json_line = json.loads(line.strip())
                    completions.append(json_line["complete_id"])


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
        sleep_countdown = 10 # Sleep the program every 10 rounds to let the CPU/GPU cool down
        for i, line in enumerate(sentences_data, start = 1):
            # Check that this line has not already been completed
            if line["id"] in completions:
                continue
            
            sentence = line["sentence"]
            words = sentence.split()
            plain_sentence = " ".join(w.rsplit("_", 1)[0] if "_" in w else w for w in words)
            adverb = line["adverb"]
            try:
                result_by_syntax = self.ablation_agents_interface.base_study(plain_sentence, adverb)
                if result_by_syntax is None:
                    continue
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
            # Record the result and the completion
            # ----------
            result = {
                "base_study": result_by_syntax,
                "kb_oneshot_cot": study_1_output,
                "kb_zeroshot": study_2_output,
                "zeroshot": study_3_output,
                "oneshot_cot": study_4_output,
                "fewshot_cot": study_5_output,
                "id": line["id"]
            }
            self.logger.log_record(result)
            self.logger.log_completion({"complete_id": line["id"]})
            sleep_countdown -= 1

            if sleep_countdown == 0:
                print("\nCoolin down for 180 seconds...\n")
                time.sleep(180)
                sleep_countdown = 10
            
