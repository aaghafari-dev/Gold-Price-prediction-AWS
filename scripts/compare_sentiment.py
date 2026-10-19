"""Score archived article metadata with the baseline lexicon and optional FinBERT.

This script does not scrape article webpages. It reads the project's normalized news
CSV, scores title + summary, and writes a second CSV that can be passed to
``scripts/train_local.py --news`` for a same-fold downstream forecast comparison.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.news import load_news
from src.features.news_features import simple_sentiment


def score_news(news: pd.DataFrame, use_finbert: bool = False) -> pd.DataFrame:
    """Return article rows with lexicon score and optional FinBERT score.

    FinBERT is optional, requires ``requirements-nlp.txt``, and may download model
    weights. The chosen ``sentiment_score`` column is what downstream aggregation
    consumes; use separate output files to compare methods fairly.
    """
    out = news.copy()
    text = (out["title"].fillna("") + " " + out.get("summary", "").fillna("")).astype(str).str.strip()
    out["lexicon_score"] = text.map(simple_sentiment)
    out["sentiment_score"] = out["lexicon_score"]
    out["sentiment_method"] = "english_keyword_lexicon"
    if use_finbert:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install optional dependencies: pip install -r requirements-nlp.txt") from exc
        classifier = pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert", top_k=None, truncation=True)
        outputs = classifier(text.tolist(), batch_size=8)
        scores, positives, negatives, neutrals, confidences, magnitudes = [], [], [], [], [], []
        for output in outputs:
            labels = {str(item["label"]).lower(): float(item["score"]) for item in output}
            pos, neg, neu = labels.get("positive", 0.0), labels.get("negative", 0.0), labels.get("neutral", 0.0)
            scores.append(pos - neg)
            positives.append(pos); negatives.append(neg); neutrals.append(neu)
            confidences.append(max(pos, neg, neu))
            magnitudes.append(pos + neg)
        out["finbert_positive_probability"] = positives
        out["finbert_negative_probability"] = negatives
        out["finbert_neutral_probability"] = neutrals
        out["sentiment_confidence"] = confidences
        out["sentiment_magnitude"] = magnitudes
        out["finbert_score"] = scores
        out["sentiment_score"] = out["finbert_score"]
        out["sentiment_method"] = "ProsusAI/finbert"
    return out


def main() -> None:
    """Parse command-line arguments, score news, and write a downstream-compatible CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news", default="data/raw/news.csv", help="Normalized timestamped article CSV")
    parser.add_argument("--out", default="data/processed/news_lexicon_scored.csv")
    parser.add_argument("--finbert", action="store_true", help="Use optional ProsusAI/finbert model")
    args = parser.parse_args()
    news = load_news(args.news)
    scored = score_news(news, use_finbert=args.finbert)
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)
    print(f"Wrote {len(scored)} scored article rows to {output}")
    print(f"Method: {scored['sentiment_method'].iloc[0] if len(scored) else 'none'}")
    print("Next: run train_local.py with --news pointing to this file; compare on identical chronological folds.")

if __name__ == "__main__":
    main()
