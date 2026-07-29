"""Search backend: lexical + vector + hybrid. Imported by agent.py and app.py."""
import json
import os
import numpy as np
from minsearch import Index, VectorSearch
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks_headerfirst.jsonl"
EMB_MODEL_NAME = "multi-qa-distilbert-cos-v1"
EMB_CACHE = f"embeddings_headerfirst_{EMB_MODEL_NAME.replace('/', '_')}.npy"

# module-level singletons so the app builds the index once, not per request
_index = None
_vindex = None
_embedding_model = None
_chunks = None


def _load():
    global _index, _vindex, _embedding_model, _chunks
    if _index is not None:
        return

    _chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            _chunks.append(json.loads(line))

    _index = Index(text_fields=["chunk"], keyword_fields=["filename"])
    _index.fit(_chunks)

    _embedding_model = SentenceTransformer(EMB_MODEL_NAME)
    if os.path.exists(EMB_CACHE):
        embeddings = np.load(EMB_CACHE)
    else:
        texts = [c["chunk"] for c in _chunks]
        embeddings = _embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
        np.save(EMB_CACHE, embeddings)

    assert embeddings.shape[0] == len(_chunks)
    assert embeddings.shape[1] == _embedding_model.encode("test").shape[0]

    _vindex = VectorSearch(keyword_fields=[])
    _vindex.fit(embeddings, _chunks)


def hybrid_search(query, num_results=5):
    _load()
    lex = _index.search(query, num_results=num_results)
    vec = _vindex.search(_embedding_model.encode(query), num_results=num_results)
    seen, merged = set(), []
    for pair in zip(lex, vec):
        for r in pair:
            k = (r["filename"], r["chunk"][:50])
            if k not in seen:
                seen.add(k); merged.append(r)
    for r in lex + vec:
        k = (r["filename"], r["chunk"][:50])
        if k not in seen:
            seen.add(k); merged.append(r)
    return merged[:num_results]


def text_search(query, num_results=5):
    return [{"filename": r["filename"], "chunk": r["chunk"]}
            for r in hybrid_search(query, num_results)]
