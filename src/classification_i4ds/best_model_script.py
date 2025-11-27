import json
import random
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from collections import Counter

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

def prepare_split(records: Iterable[dict]):
    texts = []
    labels = []
    for entry in records:
        phonemes = (entry.get("phonemes") or "").strip()
        if not phonemes:
            continue
        labels.append(int(entry["class_nr"]))
        texts.append(phonemes)
    return texts, labels

train_texts, train_labels = prepare_split(train_records)
valid_texts, valid_labels = prepare_split(valid_records)
test_texts, test_labels = prepare_split(test_records)

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

# Define vectorizers and classifiers
vectorizers = {
    "CountVectorizer": Pipeline([
        ("count", CountVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3)),
        ("scaler", Normalizer()),
        ("clf", None)
    ]),
    "TF-IDF": Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3)),
        ("clf", None)
    ])
}

classifiers = {
    "LogisticRegression": (
        LogisticRegression(max_iter=2000),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    ),
    "MultinomialNB": (
        MultinomialNB(),
        {"clf__alpha": [0.01, 0.1, 1.0]}
    ),
    "ComplementNB": (
        ComplementNB(),
        {"clf__alpha": [0.01, 0.1, 1.0]}
    ),
    "SVM": (
        LinearSVC(),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    ),
}


# For CountVectorizer, use all classifiers since with_mean=False keeps features non-negative
count_vectorizer_classifiers = list(classifiers.keys())
tfidf_classifiers = list(classifiers.keys())

# Perform grid search for each combination
best_score = -1
best_model_info = None

for vec_name, pipeline in vectorizers.items():
    if vec_name == "CountVectorizer":
        clf_list = count_vectorizer_classifiers
    else:
        clf_list = tfidf_classifiers
    
    for clf_name in clf_list:
        clf, param_grid = classifiers[clf_name]
        pipeline.set_params(clf=clf)
        
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            n_jobs=4,
            cv=3,
            refit=True,
            verbose=1,
        )
        search.fit(train_texts, train_labels)
        
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model_info = {
                "vectorizer": vec_name,
                "classifier": clf_name,
                "params": search.best_params_,
                "score": search.best_score_,
                "pipeline": search.best_estimator_
            }

print(f"Best model: {best_model_info['classifier']} with {best_model_info['vectorizer']}, params: {best_model_info['params']}, F1-macro: {best_model_info['score']}")

# Fit best model on train and evaluate on test
best_pipeline = best_model_info["pipeline"]
predictions = best_pipeline.predict(test_texts)

# Classification report
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
plt.savefig("best_model_confusion_matrix.png")
print("Confusion matrix saved to best_model_confusion_matrix.png")