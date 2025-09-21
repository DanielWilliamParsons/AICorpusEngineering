import json
from AICorpusEngineering.logger.logger import NDJSONLogger

class BroadGrouperPipeline:
    def __init__(self, grouper_agents, logger: NDJSONLogger):
        self.grouper_agents = grouper_agents
        self.logger = logger

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
                    result = self.grouper_agents.analyze_adverb(plain_sentence, adverb)
                    results.append(result)
                    validated_result = self.grouper_agents.validate(result)
                    # TODO
                    # Check that validated_result["agree"] exists first
                    # If it doesn't exist, then there was an error. Log the error
                    if validated_result["agree"] == "No":
                        mediated_result = self.grouper_agents.mediate(result, validated_result)
            self.logger.log_records(results)


        print(f"Done! Enhanced sentences saved to {output_txt}")