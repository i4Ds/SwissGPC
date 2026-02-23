# package marker for classification_i4ds
from .best_dialect_model import (
    build_best_pipeline,
    train_best_model,
    save_model,
    load_model,
    load_jsonl,
    prepare_split,
    derive_label_metadata,
)

__all__ = [
    "build_best_pipeline",
    "train_best_model",
    "save_model",
    "load_model",
    "load_jsonl",
    "prepare_split",
    "derive_label_metadata",
    "classify_dialect",  # the module itself can be imported
]
