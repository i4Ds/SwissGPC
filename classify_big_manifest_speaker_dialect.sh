#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=yt_manifest_speaker_dialect
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=performance
#SBATCH --nodes=1
#SBATCH --nodelist=calc-g-002,calc-g-004,calc-g-006
#SBATCH --output=logs/yt_manifest_speaker_dialect_%j.out
#SBATCH --error=logs/yt_manifest_speaker_dialect_%j.err

uv run python -m src.classification_i4ds.classify_manifest_speaker_dialect \
  --manifest /mnt/nas05/data02/vincenzo/podcast_data/youtube/processed/manifest.jsonl \
  --feature-source audio \
  --device auto \
  --vote-mode prob_duration
