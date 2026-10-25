"""Step 5: frozen SBERT sentence embeddings with caching."""
import hashlib
import json
import numpy as np


def embed_texts(texts, cfg, cache_path, encoder=None):
    """texts: list[str]. Returns (float32 array (n, dim), hash). Cached by text hash + checkpoint."""
    ckpt = cfg["sbert"]["checkpoint"]
    h = hashlib.sha256(("\n".join(texts) + ckpt + cfg["experiment"]["preprocessing_version"]).encode()).hexdigest()[:16]
    cache_path = cache_path.with_name(f"{cache_path.stem}_{h}.npy")
    if cache_path.exists():
        emb = np.load(cache_path)
        print("Loaded cached embeddings:", cache_path.name, emb.shape)
        return emb, h
    if encoder is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(ckpt)
    encoder.eval()
    emb = encoder.encode(texts, batch_size=cfg["sbert"]["encode_batch_size"],
                         show_progress_bar=True, convert_to_numpy=True).astype("float32")
    if emb.shape[1] != cfg["sbert"]["expected_embedding_dim"]:
        raise ValueError(f"Embedding dim {emb.shape[1]} != expected {cfg['sbert']['expected_embedding_dim']}")
    if not np.isfinite(emb).all():
        raise ValueError("NaN/Inf found in embeddings")
    np.save(cache_path, emb)
    return emb, h


def write_embedding_metadata(path, cfg, dim, n_labeled, n_unlabeled, text_hash):
    try:
        import sentence_transformers
        st_ver = sentence_transformers.__version__
    except Exception:
        st_ver = "unknown"
    meta = {"encoder_checkpoint": cfg["sbert"]["checkpoint"], "fine_tuned": False,
            "embedding_dim": int(dim), "preprocessing_version": cfg["experiment"]["preprocessing_version"],
            "sentence_transformers_version": st_ver, "n_labeled": n_labeled,
            "n_unlabeled": n_unlabeled, "text_hash": text_hash}
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
