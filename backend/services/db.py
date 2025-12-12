# backend/services/db.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import uuid
import os

class ChromaStore:
    """
    Minimal Chroma wrapper to store daily records as documents + metadata.
    We store the 'text' as document and emotions/risk as metadata.
    """

    def __init__(self, collection_name: str = "emotion_logs"):
        # Default Settings() creates an in-process DB; for production change Settings() config
        try:
            self.client = chromadb.Client(Settings())
        except Exception as e:
            raise RuntimeError("Failed to start Chroma client: " + str(e))

        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)

        # Try load a light embedding model for optional embeddings
        try:
            self.embed = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.embed = None

    def _make_doc_and_meta(self, record: Dict):
        text = record.get("text", "")
        metadata = {
            "user_id": record.get("user_id"),
            "date": record.get("date"),
            "emotions": record.get("emotions"),
            "risk_score": record.get("risk_score"),
            "risk_level": record.get("risk_level"),
        }
        return str(text), metadata

    def add_record(self, record: Dict):
        doc, meta = self._make_doc_and_meta(record)
        id_ = str(uuid.uuid4())
        if self.embed is not None:
            emb = self.embed.encode([doc])[0].tolist()
            self.collection.add(ids=[id_], documents=[doc], metadatas=[meta], embeddings=[emb])
        else:
            self.collection.add(ids=[id_], documents=[doc], metadatas=[meta])

    def get_last_n(self, user_id: str, n: int = 3) -> List[Dict]:
        # Retrieve all metadatas and filter locally (Chroma's API is limited for direct sorting by metadata)
        results = self.collection.get(include=["metadatas", "documents", "ids"])
        metas = results.get("metadatas", [])
        docs = results.get("documents", [])
        entries = []
        for meta, doc in zip(metas, docs):
            if meta and meta.get("user_id") == user_id:
                entries.append({
                    "date": meta.get("date"),
                    "emotions": meta.get("emotions") or {},
                    "risk_score": meta.get("risk_score"),
                    "risk_level": meta.get("risk_level"),
                    "doc": doc
                })
        # Sort descending by date (ISO-strings)
        entries = sorted(entries, key=lambda x: x.get("date") or "", reverse=True)
        return entries[:n]
