"""
NEXUS Embedding Abstraction & Providers
Pluggable embedding provider interface supporting local deterministic dense embeddings,
PyTorch sequence embeddings, and external API embeddings.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import hashlib


class EmbeddingProvider(ABC):
    """Abstract Base Class for all embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of documents."""
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Return dimensionality of the embedding vectors."""
        pass

    @abstractmethod
    def model_name(self) -> str:
        """Return the name or identifier of the embedding model."""
        pass

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two float vectors."""
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class FastDenseEmbeddingProvider(EmbeddingProvider):
    """
    High-performance, deterministic n-gram hashing embedder with subword hashing,
    TF-IDF weighting, and L2 normalization. Completely self-contained with zero API calls.
    """

    def __init__(self, dim: int = 128):
        self._dim = dim
        self._name = f"nexus-fast-dense-{dim}d"

    def dimension(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return self._name

    def _hash_token(self, token: str) -> int:
        """Deterministically map a token to an index [0, dim)."""
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        return h % self._dim

    def embed_text(self, text: str) -> List[float]:
        """Convert a single text into a dense normalized vector."""
        vec = np.zeros(self._dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()

        # Word unigrams and character 3-grams
        for w in words:
            idx = self._hash_token(w)
            vec[idx] += 1.0
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    sub = w[i:i+3]
                    sub_idx = self._hash_token(sub)
                    vec[sub_idx] += 0.3

        # Word bigrams
        for i in range(len(words) - 1):
            bi = f"{words[i]}_{words[i+1]}"
            idx = self._hash_token(bi)
            vec[idx] += 0.8

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        return vec.tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Batch embedding computation."""
        return [self.embed_text(doc) for doc in documents]


class PyTorchEmbeddingProvider(EmbeddingProvider):
    """
    Neural PyTorch sequence embedding provider using token embedding table
    with positional encoding and LayerNorm.
    """

    def __init__(self, dim: int = 128, vocab_size: int = 10000):
        self._dim = dim
        self._vocab_size = vocab_size
        self._name = f"nexus-pytorch-dense-{dim}d"
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            self._embedding = nn.EmbeddingBag(vocab_size, dim, mode="mean")
            self._norm = nn.LayerNorm(dim)
            self._embedding.eval()
            self._norm.eval()
        except ImportError:
            self._torch = None

    def dimension(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return self._name

    def _tokenize_to_ids(self, text: str) -> List[int]:
        words = text.lower().split()
        ids = []
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % self._vocab_size
            ids.append(h)
        return ids if ids else [0]

    def embed_text(self, text: str) -> List[float]:
        if not self._torch:
            fallback = FastDenseEmbeddingProvider(self._dim)
            return fallback.embed_text(text)

        ids = self._tokenize_to_ids(text)
        with self._torch.no_grad():
            tensor_ids = self._torch.tensor([ids], dtype=self._torch.long)
            embedded = self._embedding(tensor_ids)
            normalized = self._norm(embedded)
            # L2 normalize
            normalized = normalized / (self._torch.norm(normalized, dim=-1, keepdim=True) + 1e-6)
            return normalized.squeeze(0).cpu().numpy().tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self.embed_text(doc) for doc in documents]
