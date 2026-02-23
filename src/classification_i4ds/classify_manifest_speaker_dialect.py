"""Classify dialects for manifest entries and propagate speaker-majority labels.

Streaming behaviour:
- processes segments incrementally with tqdm
- periodically writes speaker map checkpoints to disk
- optionally writes augmented manifest at the end
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .wav2vec2_phonemes import Wav2Vec2PhonemeExtractor, normalize_phoneme_sequence

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
DEFAULT_LABEL_MAP: dict[str, str] = {
    "0": "Zürich",
    "1": "Innerschweiz",
    "2": "Wallis",
    "3": "Graubünden",
    "4": "Ostschweiz",
    "5": "Basel",
    "6": "Bern",
    "7": "Deutsch",
    "8": "Französisch",
    "9": "Italienisch",
    "10": "Englisch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict dialects for a manifest JSONL and assign a speaker dialect "
            "for each (source file, speaker) pair."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="input JSONL manifest")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent.parent / "model" / "best_dialect_model.joblib",
        help="path to trained pipeline joblib",
    )
    parser.add_argument(
        "--speaker-map-output",
        type=Path,
        default=None,
        help="output JSONL mapping file (default: <manifest>_speaker_dialects.jsonl)",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help=(
            "output JSONL with appended dialect fields "
            "(default: <manifest>_with_speaker_dialect.jsonl)"
        ),
    )
    parser.add_argument(
        "--phoneme-field",
        default="phonemes",
        help="manifest field containing phoneme text (if present, preferred input)",
    )
    parser.add_argument(
        "--audio-field",
        default="audio_path",
        help="manifest field containing per-segment audio path for wav2vec2 phonemization",
    )
    parser.add_argument(
        "--feature-source",
        choices=["auto", "phonemes", "audio"],
        default="auto",
        help="auto prefers phoneme field, then audio wav2vec2",
    )
    parser.add_argument("--speaker-field", default="speaker", help="speaker ID field")
    parser.add_argument(
        "--source-field",
        default="source_audio",
        help="field used as source key for speaker grouping",
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
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="optional JSON mapping label IDs to names, e.g. {\"0\": \"BS\"}",
    )
    parser.add_argument(
        "--vote-mode",
        choices=["prob_duration", "count"],
        default="prob_duration",
        help="prob_duration sums (probability * duration_seconds), count uses plain majority",
    )
    parser.add_argument(
        "--no-apply-to-manifest",
        action="store_true",
        help="write only speaker map; skip augmented manifest output",
    )
    return parser.parse_args()


def _default_speaker_map_path(manifest: Path) -> Path:
    return manifest.with_name(f"{manifest.stem}_speaker_dialects.jsonl")


def _default_manifest_out_path(manifest: Path) -> Path:
    return manifest.with_name(f"{manifest.stem}_with_speaker_dialect.jsonl")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} in {path}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_num} in {path} is not a JSON object")
            records.append(item)
    return records


def load_label_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return dict(DEFAULT_LABEL_MAP)
    with path.open("rt", encoding="utf-8") as infile:
        data = json.load(infile)
    if not isinstance(data, dict):
        raise ValueError(f"--label-map must point to a JSON object: {path}")
    out = dict(DEFAULT_LABEL_MAP)
    for k, v in data.items():
        out[str(k)] = str(v)
    return out


def label_name(label: str, mapping: dict[str, str]) -> str:
    return mapping.get(str(label), str(label))


def normalize_source_key(record: dict[str, Any], source_field: str) -> str:
    source = record.get(source_field)
    if source:
        return str(source)

    audio_path = record.get("audio_path", "")
    audio_path_str = str(audio_path)
    if audio_path_str:
        p = Path(audio_path_str)
        if p.suffix.lower() in AUDIO_EXTS:
            return str(p.parent)
        return audio_path_str

    return "<unknown_source>"


def majority_label(vote_counts: Counter[str], durations_by_label: dict[str, float]) -> str:
    max_votes = max(vote_counts.values())
    candidates = [label for label, c in vote_counts.items() if c == max_votes]
    if len(candidates) == 1:
        return candidates[0]
    best_duration = max(durations_by_label.get(label, 0.0) for label in candidates)
    finalists = [label for label in candidates if durations_by_label.get(label, 0.0) == best_duration]
    return sorted(finalists)[0]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("at", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_pipeline_model(path: Path):
    try:
        from joblib import load
    except ImportError as exc:
        raise RuntimeError("joblib is required to load the dialect model.") from exc
    return load(path)


def build_speaker_rows(
    grouped_state: dict[tuple[str, str], dict[str, Any]],
    vote_mode: str,
    label_map: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (source_key, speaker), st in grouped_state.items():
        vote_counts: Counter[str] = st["vote_counts"]
        durations_by_label: dict[str, float] = st["durations_by_label"]
        prob_scores: dict[str, float] = st["prob_scores"]

        if vote_mode == "prob_duration" and prob_scores:
            voted = max(prob_scores.items(), key=lambda kv: kv[1])[0]
        else:
            voted = majority_label(vote_counts, durations_by_label)

        vote_counts_sorted = dict(sorted(vote_counts.items()))
        vote_counts_named = {
            label_name(k, label_map): int(v) for k, v in vote_counts_sorted.items()
        }

        rows.append(
            {
                "source_audio": source_key,
                "speaker": speaker,
                "speaker_dialect": voted,
                "speaker_dialect_name": label_name(voted, label_map),
                "vote_mode": vote_mode,
                "num_segments": int(st["num_segments"]),
                "vote_counts": vote_counts_sorted,
                "vote_counts_named": vote_counts_named,
                "vote_scores": dict(sorted(prob_scores.items())) if prob_scores else None,
            }
        )

    rows.sort(key=lambda x: (str(x["source_audio"]), str(x["speaker"])))
    return rows


def build_speaker_rows_for_source(
    grouped_state: dict[tuple[str, str], dict[str, Any]],
    source_key: str,
    vote_mode: str,
    label_map: dict[str, str],
) -> list[dict[str, Any]]:
    subset = {k: v for k, v in grouped_state.items() if k[0] == source_key}
    rows = build_speaker_rows(subset, vote_mode, label_map)
    return rows


def main() -> None:
    args = parse_args()

    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError("tqdm is required for progress reporting.") from exc

    manifest_path = args.manifest
    speaker_map_path = args.speaker_map_output or _default_speaker_map_path(manifest_path)
    manifest_out_path = args.manifest_output or _default_manifest_out_path(manifest_path)
    label_map = load_label_map(args.label_map)

    records = load_manifest(manifest_path)
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    model = load_pipeline_model(args.model)
    if args.vote_mode == "prob_duration" and not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "vote-mode=prob_duration requires model.predict_proba; use --vote-mode count."
        )
    model_classes = [str(c) for c in getattr(model, "classes_", [])]
    if args.vote_mode == "prob_duration" and not model_classes:
        raise RuntimeError("vote-mode=prob_duration requires model.classes_.")

    t0 = time.perf_counter()

    extractor: Wav2Vec2PhonemeExtractor | None = None
    grouped_state: dict[tuple[str, str], dict[str, Any]] = {}
    segment_preds: list[str] = []
    current_source_key: str | None = None
    closed_sources: set[str] = set()

    # Start a fresh checkpoint file for this run.
    write_jsonl(speaker_map_path, [])

    for idx, rec in enumerate(tqdm(records, desc="Processing segments", unit="seg")):
        source_key = normalize_source_key(rec, args.source_field)
        if current_source_key is None:
            current_source_key = source_key
        elif source_key != current_source_key:
            rows_done = build_speaker_rows_for_source(
                grouped_state=grouped_state,
                source_key=current_source_key,
                vote_mode=args.vote_mode,
                label_map=label_map,
            )
            append_jsonl(speaker_map_path, rows_done)
            closed_sources.add(current_source_key)
            current_source_key = source_key
        if source_key in closed_sources:
            raise RuntimeError(
                f"Source key '{source_key}' appeared again after being closed. "
                "Manifest must be grouped by source file for source-level checkpoint writing."
            )

        phonemes_raw = rec.get(args.phoneme_field)
        if (
            args.feature_source in ("auto", "phonemes")
            and isinstance(phonemes_raw, str)
            and phonemes_raw.strip()
        ):
            feature = normalize_phoneme_sequence(phonemes_raw.strip())
        else:
            audio_path_raw = rec.get(args.audio_field)
            if not (isinstance(audio_path_raw, str) and audio_path_raw.strip()):
                raise ValueError(f"Record {idx} missing '{args.audio_field}'")
            audio_path = audio_path_raw.strip()
            if not Path(audio_path).exists():
                raise ValueError(f"Record {idx} missing audio file: {audio_path}")
            if extractor is None:
                extractor = Wav2Vec2PhonemeExtractor(
                    model_name=args.wav2vec2_model,
                    device=args.device,
                )
            feature = extractor.phonemize_audio_path(audio_path)

        pred = str(model.predict([feature])[0])
        segment_preds.append(pred)

        duration_val = rec.get("duration", 0.0)
        try:
            duration = float(duration_val)
        except (TypeError, ValueError):
            duration = 0.0

        speaker = str(rec.get(args.speaker_field, "")).strip() or "<unknown_speaker>"
        gk = (source_key, speaker)
        if gk not in grouped_state:
            grouped_state[gk] = {
                "vote_counts": Counter(),
                "durations_by_label": defaultdict(float),
                "prob_scores": defaultdict(float),
                "num_segments": 0,
            }

        st = grouped_state[gk]
        st["vote_counts"][pred] += 1
        st["durations_by_label"][pred] += duration
        st["num_segments"] += 1

        if args.vote_mode == "prob_duration":
            probs = model.predict_proba([feature])[0]
            for cls, p in zip(model_classes, probs):
                st["prob_scores"][str(cls)] += float(p) * duration

    if current_source_key is not None:
        rows_done = build_speaker_rows_for_source(
            grouped_state=grouped_state,
            source_key=current_source_key,
            vote_mode=args.vote_mode,
            label_map=label_map,
        )
        append_jsonl(speaker_map_path, rows_done)
        closed_sources.add(current_source_key)

    speaker_map_rows = build_speaker_rows(grouped_state, args.vote_mode, label_map)

    manifest_written = False
    if not args.no_apply_to_manifest:
        majority_by_group = {
            (row["source_audio"], row["speaker"]): str(row["speaker_dialect"])
            for row in speaker_map_rows
        }
        augmented_rows: list[dict[str, Any]] = []
        for idx, rec in enumerate(records):
            speaker = str(rec.get(args.speaker_field, "")).strip() or "<unknown_speaker>"
            source_key = normalize_source_key(rec, args.source_field)
            group_key = (source_key, speaker)

            out = dict(rec)
            segment_label = segment_preds[idx]
            majority_label_id = majority_by_group[group_key]
            out["dialect_segment"] = segment_label
            out["dialect_segment_name"] = label_name(segment_label, label_map)
            out["dialect_speaker_majority"] = majority_label_id
            out["dialect_speaker_majority_name"] = label_name(majority_label_id, label_map)
            augmented_rows.append(out)

        write_jsonl(manifest_out_path, augmented_rows)
        manifest_written = True

    elapsed = time.perf_counter() - t0
    speed = len(records) / elapsed if elapsed > 0 else 0.0

    print(f"Input records: {len(records)}")
    print(f"Unique (source, speaker) groups: {len(speaker_map_rows)}")
    print(f"Speaker dialect map written to: {speaker_map_path}")
    if manifest_written:
        print(f"Augmented manifest written to: {manifest_out_path}")
    else:
        print("Augmented manifest writing skipped (--no-apply-to-manifest).")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Throughput segments/sec: {speed:.2f}")


if __name__ == "__main__":
    main()
