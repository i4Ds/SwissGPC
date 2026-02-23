"""Command line dialect classifier.

This script loads the trained pipeline stored under ``src/model/best_dialect_model.joblib``
and applies it to raw audio files. The current best pipeline is a
``CountVectorizer`` over phoneme symbols, so inputs are first converted to phonemes
using a wav2vec2 phoneme model.

Usage:

```sh
# classify a list of wav files and write CSV
python -m classification_i4ds.classify_dialect audio1.wav audio2.wav \
    --output results.csv
```

The script prints a tab‑separated summary to stdout by default and can also save a
CSV file when ``--output`` is given.  You may specify a different model with
``--model``; by default it points to the joblib produced by the training code.
"""

import argparse
from pathlib import Path
from typing import List

# import local utilities
from .best_dialect_model import load_model
from .wav2vec2_phonemes import Wav2Vec2PhonemeExtractor


def classify_paths(paths: List[Path], model, extractor: Wav2Vec2PhonemeExtractor) -> List[str]:
    """Run the pipeline on audio paths, returning predictions."""
    texts = [extractor.phonemize_audio_path(p) for p in paths]
    return model.predict(texts)


def main():
    parser = argparse.ArgumentParser(
        description="Dialect classifier for audio files via wav2vec2 phoneme extraction"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="audio files (wav/mp3/flac/...)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent.parent / "model" / "best_dialect_model.joblib",
        help="path to trained pipeline joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional output CSV path",
    )
    parser.add_argument(
        "--wav2vec2-model",
        default="facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        help="HuggingFace wav2vec2 phoneme model ID/path",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="inference device for wav2vec2 phoneme extraction",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    extractor = Wav2Vec2PhonemeExtractor(model_name=args.wav2vec2_model, device=args.device)
    preds = classify_paths([Path(p) for p in args.inputs], model, extractor)

    for inp, pred in zip(args.inputs, preds):
        print(f"{inp}\t{pred}")

    if args.output:
        import csv

        with args.output.open("w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(["input", "prediction"])
            writer.writerows(zip(args.inputs, preds))


if __name__ == "__main__":
    main()
