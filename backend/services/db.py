# backend/services/db.py

import json
import os
from typing import List, Dict


class JsonStore:
    def __init__(self, file_path: str = "data/emotion_logs.json"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        if not os.path.exists(self.file_path):
            self._save_all([])

    # ------------------------
    # Internal helpers
    # ------------------------

    def _load_all(self) -> List[Dict]:
        if not os.path.exists(self.file_path):
            return []

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

    # ------------------------
    # Public APIs
    # ------------------------

    def add_record(self, record: Dict):
        data = self._load_all()
        data.append(record)
        self._save_all(data)

    def get_last_n(self, employee_id: str, n: int = 2) -> List[Dict]:
        data = self._load_all()

        filtered = [
            r for r in data
            if r.get("employee_id") == employee_id
        ]

        filtered.sort(key=lambda x: x.get("date", ""), reverse=True)
        return filtered[:n]

    def get_records_by_org(self, org_id: str) -> List[Dict]:
        data = self._load_all()

        return [
            r for r in data
            if r.get("org_id") == org_id
            and "employee_id" in r
            and "wellbeing_score" in r
        ]
