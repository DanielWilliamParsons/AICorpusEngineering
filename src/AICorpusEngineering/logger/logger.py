import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class NDJSONLogger:
    """
    Append-only logger that writes each record as one JSON object per line (NDJSON format).
    """

    def __init__(self, data_logs: Path | None, error_logs: Path | None, output_dir: Path):
        """
        log_file: user-supplied path (can be None, a file path, or a directory)
        output_dir: the run's output directory (used for default logs)
        """

        if error_logs is None:
            # Default: put a timestamped log file inside output_dir
            error_logs = output_dir / f"errors_{datetime.now().strftime('%Y%m%d-%H%M%S')}.ndjson"
        else:
            error_logs = Path(error_logs).expanduser().resolve()

            if error_logs.suffix == "":
                # User passed a directory -> put a timestamped log file inside it
                error_logs.mkdir(parents=True, exist_ok=True)
                error_logs = error_logs / f"errors_{datetime.now().strftime('%Y%m%d-%H%M%S')}.ndjson"
            else:
                # User passed a file path -> ensure its parent exists
                error_logs.parent.mkdir(parent=True, exist_ok=True)

        self.error_logs = error_logs

        if data_logs is None:
            # Default: put a timestamped log file inside output_dir
            data_logs = output_dir / f"data_{datetime.now().strftime('%Y%m%d-%H%M%S')}.ndjson"
        else:
            data_logs = Path(error_logs).expanduser().resolve()

            if data_logs.suffix == "":
                # User passed a directory -> put a timestamped log file inside it
                data_logs.mkdir(parents=True, exist_ok=True)
                data_logs = data_logs / f"data_{datetime.now().strftime('%Y%m%d-%H%M%S')}.ndjson"
            else:
                # User passed a file path -> ensure its parent exists
                data_logs.parent.mkdir(parent=True, exist_ok=True)

        self.data_logs = data_logs

    def log_error(self, error_record):
        """
        Append an error record to the log file, located in the global logger instance
        """
        with self.error_logs.open("a", encoding="utf-8") as f:
            f.write(json.dumps(error_record, ensure_ascii=False) + "\n")

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