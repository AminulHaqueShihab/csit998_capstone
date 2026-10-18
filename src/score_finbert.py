"""Stage 8: score the cleaned headlines with FinBERT.

Run: python -m src.score_finbert
"""
import json
import time

import numpy as np
import pandas as pd
import torch

from . import config
from .config import PipelineError

CHUNK = 20_000
BATCH = 64
MAX_TOKENS = 64  # headlines are short; longer ones are truncated


def score_texts(texts, tokenizer, model, device) -> np.ndarray:
    """Return (n, 3) softmax probabilities [positive, negative, neutral] in the input order."""
    order = np.argsort([len(t) for t in texts])  # similar lengths per batch = less padding
    out = np.zeros((len(texts), 3), dtype="float32")
    with torch.no_grad():
        for i in range(0, len(order), BATCH):
            idx = order[i: i + BATCH]
            enc = tokenizer([texts[j] for j in idx], padding=True, truncation=True, max_length=MAX_TOKENS,
                            return_tensors="pt").to(device)
            out[idx] = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
    return out


def main():
    config.require_file(config.NEWS_CLEAN, "score_finbert", "Run `python -m src.prepare_news` first.")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    news = pd.read_csv(config.NEWS_CLEAN, usecols=["article_id", "headline"], dtype=str)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.FINBERT_MODEL).eval().to(device)
    labels = {int(k): v for k, v in model.config.id2label.items()}
    if [labels[i] for i in range(3)] != ["positive", "negative", "neutral"]:
        raise PipelineError(f"[score_finbert] unexpected label order {labels}")

    config.FINBERT_PARTS.mkdir(parents=True, exist_ok=True)
    texts = news["headline"].tolist()
    n_chunks = (len(texts) + CHUNK - 1) // CHUNK
    t0 = time.time()
    for c in range(n_chunks):
        part = config.FINBERT_PARTS / f"part_{c:03d}.npy"
        if part.exists():
            continue
        np.save(part, score_texts(texts[c * CHUNK: (c + 1) * CHUNK], tokenizer, model, device))
        print(f"[score_finbert] chunk {c + 1}/{n_chunks} done ({time.time() - t0:.0f}s)", flush=True)

    probs = np.concatenate([np.load(config.FINBERT_PARTS / f"part_{c:03d}.npy") for c in range(n_chunks)])
    if len(probs) != len(news):
        raise PipelineError(f"[score_finbert] scored {len(probs)} rows but expected {len(news)}")
    out = pd.DataFrame({"article_id": news["article_id"], "finbert_pos": probs[:, 0], "finbert_neg": probs[:, 1],
                        "finbert_neu": probs[:, 2]})
    out["finbert_score"] = out["finbert_pos"] - out["finbert_neg"]
    out["finbert_label"] = np.array(["positive", "negative", "neutral"])[probs.argmax(axis=1)]
    out.to_csv(config.NEWS_FINBERT, index=False)
    config.write_report("finbert_report.json", {
        "model": config.FINBERT_MODEL, "model_commit": getattr(model.config, "_commit_hash", None),
        "device": device, "scored_headlines": len(out), "max_tokens": MAX_TOKENS,
        "label_counts": out["finbert_label"].value_counts().to_dict(),
        "score_mean": float(out["finbert_score"].mean()),
        "note": "Device and library versions can change the 6th decimal of probabilities, rarely a label.",
        "output": str(config.NEWS_FINBERT.relative_to(config.ROOT)),
    })
    print(f"[score_finbert] {len(out)} headlines scored; labels {out['finbert_label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
