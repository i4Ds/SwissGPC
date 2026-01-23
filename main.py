import argparse
import os

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _process_podcast_steps(
    podcast_name: str,
    source: str,
    config: dict,
    youtube_url: str | None = None,
) -> None:
    from src.classification.dialect_classifier import (
        dialect_identification_naive_bayes_majority_voting,
    )
    from src.download.download_from_srf import (
        download_srf_podcast_audio,
        download_srf_podcast_metadata,
    )
    from src.download.download_from_yt import (
        download_yt_podcast_audio,
        download_yt_podcast_metadata,
    )
    from src.processing.move_audio_to_dialect import move_podcast_to_dialect
    from src.segmentation.segmentation import diarize_and_segment_podcast
    from src.synthesis.mel_spectrogram import create_mel_spectrogram
    from src.transcription.transcribe_to_phoneme import audio_to_phoneme
    from src.transcription.transcribe_to_swiss_german import transcribe_de_to_ch

    write_to_hdf5 = config["write_attrs_to_hdf5"]
    logger.info(f"Transcribing Podcast {podcast_name} from {source}.")

    # Step 1: Download Audio
    if config["steps"]["download"]:
        if source == "youtube":
            if not youtube_url:
                raise ValueError(f"Missing youtube_url for podcast '{podcast_name}'")
            download_yt_podcast_metadata(podcast_name, youtube_url)
            download_yt_podcast_audio(podcast_name)
        else:
            download_srf_podcast_metadata(podcast_name)
            download_srf_podcast_audio(podcast_name)

    # Step 2: Speaker Diarization & German Transcription & Segmentation
    if config["steps"]["diarization"] or config["steps"]["segmentation"]:
        diarize_and_segment_podcast(
            podcast_name,
            config["steps"]["diarization"],
            config["steps"]["segmentation"],
            copy_to_projects=True,
        )

    # Step 3: Phoneme Transcription
    if config["steps"]["phon_transcription"]:
        audio_to_phoneme(
            podcast_name,
            write_to_hdf5,
            overwrite_existing_samples=False,
            copy_from_projects=True,
        )

    # Step 4: Dialect Identification
    if config["steps"]["dialect_classification"]:
        dialect_identification_naive_bayes_majority_voting(podcast_name)

    # Step 5: Swiss German Text Generation
    if config["steps"]["ch_transcription"]:
        transcribe_de_to_ch(podcast_name, write_to_hdf5)

    # Step 6: Mel-Spectrogram Generation
    if config["steps"]["mel_spectrogram"]:
        create_mel_spectrogram(podcast_name)

    # Step 7: Move podcast based h5 into central dialect h5
    if config["steps"]["move_into_dialect_h5"]:
        move_podcast_to_dialect(podcast_name)


def main(config_path):
    logger.info("Started")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_root = config.get("data_root")
    if data_root:
        os.environ["PODCAST_DATA_ROOT"] = data_root

    source = config.get("source", "").lower()
    if source not in ["srf", "youtube"]:
        raise ValueError("config 'source' must be either 'srf' or 'youtube'")

    podcasts = config.get("podcasts", [])
    if not podcasts:
        logger.warning("No podcasts configured in config.")

    if source == "srf":
        for podcast_name in podcasts:
            try:
                _process_podcast_steps(podcast_name, "srf", config)
            except Exception:
                logger.error("Failed to process SRF podcast %s", podcast_name, exc_info=True)
                continue
    else:
        for entry in podcasts:
            podcast_name = entry.get("title") or entry.get("name")
            youtube_url = entry.get("url")
            if not podcast_name:
                raise ValueError("YouTube podcast entry is missing 'title'")
            try:
                _process_podcast_steps(podcast_name, "youtube", config, youtube_url=youtube_url)
            except Exception:
                logger.error("Failed to process YouTube podcast %s", podcast_name, exc_info=True)
                continue

    logger.info("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
