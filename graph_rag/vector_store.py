import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any

class VectorStore:
    def __init__(self):
        self.embeddings: Dict[str, np.ndarray] = {}  # key -> numpy array
        self.texts: Dict[str, str] = {}              # key -> raw text chunk
        self.metadata: Dict[str, dict] = {}          # key -> metadata dict

    def add_item(self, key: str, text: str, embedding: List[float], metadata: Dict[str, Any] = None) -> None:
        """Adds a document chunk and its embedding to the store."""
        vec = np.array(embedding, dtype=np.float32)
        # Normalize the vector for faster cosine similarity via dot product
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self.embeddings[key] = vec
        self.texts[key] = text
        self.metadata[key] = metadata or {}

    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Performs a fast cosine similarity search using NumPy operations."""
        if not self.embeddings:
            return []
            
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
            
        # Get list of keys and construct matrix
        keys = list(self.embeddings.keys())
        matrix = np.vstack([self.embeddings[key] for key in keys])
        
        # Cosine similarity (since vectors are normalized, dot product is cosine similarity)
        similarities = np.dot(matrix, q_vec)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            key = keys[idx]
            score = float(similarities[idx])
            results.append((key, self.texts[key], score, self.metadata[key]))
            
        return results

    def save(self, filepath: Path) -> None:
        """Serializes the vector store to a file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert NumPy arrays to lists for JSON serialization
        serializable_embeddings = {k: v.tolist() for k, v in self.embeddings.items()}
        data = {
            "embeddings": serializable_embeddings,
            "texts": self.texts,
            "metadata": self.metadata
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: Path) -> None:
        """Loads vector store from a serialized JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            return
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.embeddings = {k: np.array(v, dtype=np.float32) for k, v in data["embeddings"].items()}
        self.texts = data["texts"]
        self.metadata = data["metadata"]
        
    def __len__(self) -> int:
        return len(self.embeddings)
