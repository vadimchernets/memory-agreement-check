"""Example adapter: build memory contexts for LoCoMo with FlyHash (or SimHash / dense cosine) and write the
JSONL that `python -m memcheck run --contexts ...` reads.

    git clone https://github.com/mohanji/flyhash-agent-memory
    pip install numpy sentence-transformers
    python examples/flyhash_contexts.py --flyhash-src flyhash-agent-memory/src \
        --method flyhash --active-bits 32 --k 10 --out contexts/flyhash-k32.jsonl
    python -m memcheck run --backend cli --contexts contexts/flyhash-k32.jsonl --memory-name flyhash-k32

Each line written: {"qid": "conv-26/0", "memories": [{"text": "...", "dia_id": "D1:3", "score": 12.0}, ...]}
Keeping `dia_id` lets memcheck tell exactly whether the evidence turn was retrieved.

FlyHash is by Bheemadolu & Odhiambo (arXiv:2611.05791, MIT). This file only calls its public FlyHash/SimHash
classes; it does not reproduce their benchmark. `--embedder bow` is a crude bag-of-words stand-in for smoke tests.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memcheck import locomo  # noqa: E402


def embed(texts, how: str):
    if how == "minilm":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return np.asarray(m.encode(texts, normalize_embeddings=True, batch_size=64), dtype=np.float32)
    # bow: hashed bag of words into 384 dims (no download; only for trying the pipeline)
    import zlib
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % 384] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flyhash-src", help="path to flyhash-agent-memory/src (needed for flyhash/simhash)")
    ap.add_argument("--method", choices=["flyhash", "simhash", "dense"], default="flyhash")
    ap.add_argument("--active-bits", type=int, default=32, help="FlyHash active bits (k); SimHash uses this many bits")
    ap.add_argument("--hash-dim", type=int, default=2000, help="FlyHash expansion size")
    ap.add_argument("--k", type=int, default=10, help="memories returned per question")
    ap.add_argument("--embedder", choices=["minilm", "bow"], default="minilm")
    ap.add_argument("--data", help="locomo10.json (default: memcheck cache)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    convs, qs = locomo.load(Path(a.data) if a.data else None)
    if a.method != "dense":
        sys.path.insert(0, a.flyhash_src or ".")
        from flyhash import FlyHash, SimHash

    out = open(a.out, "w")
    for sid, turns in convs.items():
        T = embed([t.search_text() for t in turns], a.embedder)
        sq = [q for q in qs if q.sample_id == sid]
        Q = embed([q.question for q in sq], a.embedder)
        if a.method == "flyhash":
            h = FlyHash(input_dim=T.shape[1], hash_dim=a.hash_dim, wta_percent=a.active_bits / a.hash_dim, seed=0)
            ct, cq = h.hash(T).astype(np.int32), h.hash(Q).astype(np.int32)
            S = cq @ ct.T                                   # shared active bits
        elif a.method == "simhash":
            h = SimHash(input_dim=T.shape[1], hash_dim=a.active_bits, seed=0)
            ct, cq = h.hash(T).astype(np.int32), h.hash(Q).astype(np.int32)
            S = a.active_bits - (cq[:, None, :] != ct[None, :, :]).sum(-1)   # bits in agreement
        else:
            S = Q @ T.T                                     # exact cosine (upper bound for an ANN index)
        for qi, q in enumerate(sq):
            order = sorted(range(len(turns)), key=lambda i: (-S[qi, i], i))[: a.k]
            order.sort()                                    # chronological, like the built-in memory
            mems = [{"text": turns[i].render(), "dia_id": turns[i].dia_id, "score": float(S[qi, i])} for i in order]
            out.write(json.dumps({"qid": q.qid, "memories": mems}) + "\n")
    out.close()
    print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
