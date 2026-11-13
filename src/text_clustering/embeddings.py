"""Offline fixture encoder plus explicit opt-in model loading."""
import hashlib
import numpy as np
from .vectors import unit_rows


class HashingEmbedder:
    """Deterministic character n-gram baseline, NOT a semantic embedding model."""
    def __init__(self, dimensions=512):
        if dimensions < 2:
            raise ValueError("dimensions must be >= 2")
        self.dimensions = dimensions
        self.identity = f"char-bigram-sha256-v1:{dimensions}"

    def encode(self, texts):
        x = np.zeros((len(texts), self.dimensions))
        for row, text in enumerate(texts):
            text = text.strip().casefold()
            if not text:
                raise ValueError("empty text")
            grams = [text[i:i + 2] for i in range(max(1, len(text) - 1))]
            for gram in grams:
                h = int.from_bytes(hashlib.sha256(gram.encode()).digest()[:8], "big")
                x[row, h % self.dimensions] += 1
        return unit_rows(x)


class SentenceTransformerEmbedder:
    """Loads only when constructed; no implicit checkpoint choice/download."""
    def __init__(self, model_name, revision, batch_size=32, device="cpu",
                 local_files_only=True, prompt=None):
        if not revision:
            raise ValueError("pin model revision")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, revision=revision, device=device,
                                         local_files_only=local_files_only)
        self.batch_size, self.prompt = batch_size, prompt
        self.identity = f"{model_name}@{revision};prompt={prompt!r};normalized"

    def encode(self, texts):
        return unit_rows(self.model.encode(list(texts), batch_size=self.batch_size,
                         prompt=self.prompt, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False))
