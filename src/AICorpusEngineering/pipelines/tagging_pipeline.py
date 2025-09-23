import json
from AICorpusEngineering.logger.logger import NDJSONLogger
from AICorpusEngineering.error_handler.error_handler import error_handler
from AICorpusEngineering.logger.logger_registry import get_logger

class TaggingPipeline:
    def __init__(self, grouper_agents, logger: NDJSONLogger):
        self.grouper_agents = grouper_agents
        self.logger = get_logger() # Get the global instance of the logger

    def run(self, input_txt, output_txt):
        results = []
        with open(input_txt, "r", encoding="utf-8") as infile:
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
                            results.append(result_by_syntax)
                    except Exception as e:
                        if error_handler:
                            # TODO: add file name to the context
                            error_handler.handle(e, context={"line": i, "sentence": plain_sentence, "adverb": adverb}) # Logging of the error is handled by the error_handler so no need to log
            self.logger.log_records(results)


        print(f"Done! Enhanced sentences saved to {output_txt}")