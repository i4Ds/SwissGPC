import json
import random
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from collections import Counter
from joblib import dump, load

# Set paths
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "src" / "dialect_data"

# Reproducible seeds
random.seed(144333)
np.random.seed(144333)

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("rt", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def derive_label_metadata(*datasets):
    mapping = {}
    for dataset in datasets:
        for entry in dataset:
            if "class_nr" not in entry:
                continue
            label = int(entry["class_nr"])
            name = str(entry.get("classname", label))
            mapping[label] = name
    if not mapping:
        raise ValueError("Could not derive any labels.")
    ordered = sorted(mapping)
    names = [mapping[label] for label in ordered]
    return ordered, names

def prepare_split(records: Iterable[dict]):
    texts = []
    labels = []
    speakers = []
    for entry in records:
        phonemes = (entry.get("phonemes") or "").strip()
        if not phonemes:
            continue
        labels.append(int(entry["class_nr"]))
        texts.append(phonemes)
        speakers.append(entry.get("speaker", ""))
    return texts, labels, speakers

def build_best_pipeline():
    return Pipeline([
        ("count", CountVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3)),
        ("scaler", Normalizer()),
        ("clf", LogisticRegression(C=10.0, max_iter=2000))
    ])

def train_best_model(train_texts, train_labels):
    pipeline = build_best_pipeline()
    pipeline.fit(train_texts, train_labels)
    return pipeline

def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
    print(f"Model saved to {path}")

def load_model(path: Path):
    model = load(path)
    print(f"Model loaded from {path}")
    return model

if __name__ == "__main__":
    # Load SNF datasets (assuming Swiss German dialects)
    train_records = load_jsonl(DATA_DIR / "snf_train_train_all.jsonl")
    valid_records = load_jsonl(DATA_DIR / "snf_train_valid.jsonl")
    test_records = load_jsonl(DATA_DIR / "snf_test_test.jsonl")

    print(f"Train: {len(train_records)} records")
    print(f"Valid: {len(valid_records)} records")
    print(f"Test: {len(test_records)} records")

    # Class distribution and labels
    all_records = train_records + valid_records + test_records
    class_counts = {}
    for rec in all_records:
        cls = rec.get("class_nr")
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print("Class distribution:", class_counts)

    label_ids, label_names = derive_label_metadata(train_records, valid_records, test_records)
    print(f"Label IDs: {label_ids}")
    print(f"Label names: {label_names}")

    train_texts, train_labels, _ = prepare_split(train_records)
    valid_texts, valid_labels, _ = prepare_split(valid_records)
    test_texts, test_labels, _ = prepare_split(test_records)

    print(f"Train: {len(train_texts)} texts")
    print(f"Valid: {len(valid_texts)} texts")
    print(f"Test: {len(test_texts)} texts")

    # Preprocess texts: remove spaces
    train_texts = [text.replace(" ", "") for text in train_texts]
    valid_texts = [text.replace(" ", "") for text in valid_texts]
    test_texts = [text.replace(" ", "") for text in test_texts]

    print("Spaces removed from texts.")

    # Phoneme frequency distribution
    all_train_text = ''.join(train_texts)
    phoneme_counts = Counter(all_train_text)

    # Filter out rare phonemes (appear < 100 times)
    rare_phonemes = {phoneme for phoneme, count in phoneme_counts.items() if count < 100}
    print(f"Removing {len(rare_phonemes)} rare phonemes.")

    train_texts = [''.join(c for c in text if c not in rare_phonemes) for text in train_texts]
    valid_texts = [''.join(c for c in text if c not in rare_phonemes) for text in valid_texts]
    test_texts = [''.join(c for c in text if c not in rare_phonemes) for text in test_texts]

    print("Rare phonemes removed.")

    # Train the best model
    model = train_best_model(train_texts, train_labels)

    # Evaluate on test
    predictions = model.predict(test_texts)
    report = classification_report(test_labels, predictions, output_dict=True)
    print("Classification Report for Best Model:")
    print(pd.DataFrame(report).transpose())

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        test_labels, predictions, normalize="true", cmap="Blues", ax=ax,
        display_labels=label_names, labels=label_ids, xticks_rotation="vertical"
    )
    fig.tight_layout()
    plt.title("Best Model Test Confusion Matrix")
    plt.savefig("best_dialect_model_confusion_matrix.png")
    print("Confusion matrix saved to best_dialect_model_confusion_matrix.png")

    # Save model
    save_model(model, PROJECT_ROOT / "src" / "model" / "best_dialect_model.joblib")