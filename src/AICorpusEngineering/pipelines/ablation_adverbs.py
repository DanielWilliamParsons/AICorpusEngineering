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
    
    def run(self, input_dir):

        # ----------
        # Determine which files are to be covered
        # TODO: Currently this processes tagged text files
        # However, in future, the gold standard sentences will be used
        # So there will likely be just ONE file in which the gold standard sentences will be kept
        # Update later to process just that one file and loop through each sentence
        # ----------
        completed_files = []
        logs_dir = self.logger.logs_dir # This is set when initializing the logger and indicates where the data can be stored

        for run_completion in logs_dir.glob("_run_completion_*.ndjson"):
            with open(run_completion, "r", encoding="utf-8") as infile:
                for i, line in enumerate(infile, start=1):
                    json_line = json.loads(line.strip())
                    completed_files.append(json_line["filepath"])
        print(f"The completed files are {completed_files}")


        # ----------
        # Process individual files
        # ----------
        for input_file in input_dir.glob("*.txt"):
            results = []
            filename = input_file.name

            if str(input_file) in completed_files:
                continue

            with open(input_file, "r", encoding="utf-8") as infile:
                for i, line in enumerate(infile, start=1):
                    sentence = line.strip()
                    if not sentence:
                        continue

                    words = sentence.split()
                    adverbs = [w.rsplit("_", 1)[0] for w in words if "_" in w and w.rsplit("_")[1] in ["ADV"]]
                    plain_sentence = " ".join(w.rsplit("_", 1)[0] if "_" in w else w for w in words)

                    # Loop through each adverb in adverbs and send to ablation_agents_interface for analysis
                    for adverb in adverbs:
                        try:
                            # TODO: Update how the output are handled and saved to the data logs
                            result_by_syntax = self.ablation_agents_interface.base_study(plain_sentence, adverb)
                            study_1_output = self.ablation_agents_interface.kb_oneshot_cot(plain_sentence, adverb)
                            study_2_output = self.ablation_agents_interface.kb_zeroshot(plain_sentence, adverb)
                            study_3_output = self.ablation_agents_interface.zeroshot(plain_sentence, adverb)
                            study_4_output = self.ablation_agents_interface.oneshot_cot(plain_sentence, adverb)
                            study_5_output = self.ablation_agents_interface.fewshot_cot(plain_sentence, adverb)
                            if result_by_syntax:
                                result = {"filename": filename, "result": result_by_syntax}
                                # Store the result in an array
                                results.append(result)
                        except Exception as e:
                            if error_handler:
                                error_handler.handle(e, context={"filename": filename, "line": i, "sentence": plain_sentence, "adverb": adverb}) # Logging of the error is handled by the error_handler so no need to log
            
            # ----------
            # Update the completion log and data records
            # ----------
            completion_log = {"filepath": str(input_file)}
            self.logger.log_completion(completion_log)
            for result in results:
                self.logger.log_record(result)