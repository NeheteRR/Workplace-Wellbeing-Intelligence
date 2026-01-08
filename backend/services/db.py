import json
import os
from typing import Dict, List

class JsonStore:
    def __init__(self, file_path: str = "data/emotion_logs.json"):
        self.file_path = file_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # Ensure file exists and is valid JSON
        if not os.path.exists(self.file_path):
            self._save_all([])
        else:
            # Handle empty or corrupted file
            try:
                self._load_all()
            except Exception:
                self._save_all([])

    def _load_all(self) -> List[Dict]:
        if not os.path.exists(self.file_path):
            return []

        # Handle empty file safely
        if os.path.getsize(self.file_path) == 0:
            return []

        with open(self.file_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def _save_all(self, data: List[Dict]):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_record(self, record: Dict):
        data = self._load_all()
        data.append(record)
        self._save_all(data)



    def get_last_n(self, user_id: str, n: int = 3) -> List[Dict]:
        data = self._load_all()

        user_data = [
            d for d in data
            if d.get("user_id") == user_id
        ]

        user_data.sort(key=lambda x: x.get("date", ""), reverse=True)
        return user_data[:n]
