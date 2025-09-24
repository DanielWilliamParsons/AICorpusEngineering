import json
from AICorpusEngineering.logger.logger import NDJSONLogger
from AICorpusEngineering.error_handler.error_handler import error_handler
from AICorpusEngineering.logger.logger_registry import get_logger
from pathlib import Path

class TaggingPipeline:
    def __init__(self, grouper_agents, logger: NDJSONLogger):
        self.grouper_agents = grouper_agents
        self.logger = get_logger() # Get the global instance of the logger

    def run(self, input_dir, output_dir):
        # Find the _run_completion logs to know which files should be excluded
        # They might be in a user defined directory
        completed_files = []
        logs_dir = self.logger.logs_dir # This is set when initializing the logger and indicates where the data can be stored

        for run_completion in logs_dir.glob("_run_completion_*.ndjson"):
            with open(run_completion, "r", encoding="utf-8") as infile:
                for i, line in enumerate(infile, start=1):
                    json_line = json.loads(line.strip())
                    completed_files.append(json_line["filepath"])
        print(f"The completed files are {completed_files}")

        # Loop through all the files in the input_dir
        for input_file in input_dir.glob("*.txt"):
            results = []
            # First make sure the file has not already been processed, and skip if it has.
            filename = input_file.name
            print(f"The filename being explored is: {str(input_file)}")
            if str(input_file) in completed_files:
                continue

            with open(input_file, "r", encoding="utf-8") as infile:
                for i, line in enumerate(infile, start=1):
                    sentence = line.strip()
                    if not sentence:
                        continue

                    # Extract tokens and adverbs and make a plain, untagged sentence
                    words = sentence.split()
                    adverbs = [w.rsplit("_", 1)[0] for w in words if "_" in w and w.rsplit("_")[1] in ["ADV"]]
                    plain_sentence = " ".join(w.rsplit("_", 1)[0] if "_" in w else w for w in words)

                    # Loop through each adverb in adverbs and send to grouper_agents for analysis
                    for adverb in adverbs:
                        try:
                            result_by_syntax = self.grouper_agents.analyze_by_syntax(plain_sentence, adverb)
                            if result_by_syntax:
                                result = {"filename": filename, "result": result_by_syntax}
                                # Store the result in an array
                                results.append(result)
                        except Exception as e:
                            if error_handler:
                                # TODO: add file name to the context
                                error_handler.handle(e, context={"filename": filename, "line": i, "sentence": plain_sentence, "adverb": adverb}) # Logging of the error is handled by the error_handler so no need to log
            completion_log = {"filepath": str(input_file)}
            
            # Log all the results from this file's run
            for result in results:
                self.logger.log_record(result)
            
            # Log the completion of the run
            self.logger.log_completion(completion_log)

        print(f"Done! Enhanced sentences saved to {output_dir}")