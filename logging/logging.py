import json
from typing import Dict, Any, List


class NDJSONLogger:
    """
    Append-only logger that writes each record as one JSON object per line (NDJSON format).
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    def log_record(self, record: Dict[str, Any]) -> None:
        """
        Append a single record to the log file as a JSON line.
        """
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_records(self, records: List[Dict[str, Any]]) -> None:
        """
        Append multiple records (list of dicts) to the log file.
        """
        with open(self.filepath, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_all(self) -> List[Dict[str, Any]]:
        """
        Load all records from the log file into memory as a list of dicts.
        """
        results = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # skip empty lines
                    results.append(json.loads(line))
        return results