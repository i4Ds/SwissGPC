import os
import time
import pandas as pd

from pytubefix import Playlist, YouTube
from pytubefix.exceptions import RegexMatchError, VideoUnavailable
from pytubefix.extract import video_id
from pytubefix.helpers import reset_cache

from src.download.utils import get_downloaded_metadata, save_podcast_metadata_to_csv, logger, \
    load_podcast_metadata_from_csv, create_audio_folder_if_not_exists, get_podcast_path

new_podcasts = [
    {"title": "SRF Dokumentationen", "url": "https://www.youtube.com/playlist?list=PLrAvDZ9sYjXYQb1Jk4TSyy6JTXbDn1JLg",
     "lang": "mixed"},
    {"title": "SRF Reportagen", "url": "https://www.youtube.com/playlist?list=PLrAvDZ9sYjXZ72dR3c-xdnAjRIVmTIMIG",
     "lang": "mixed"}
]

existing_podcasts = [
    {"title": "Sexologie - Wissen macht Lust",
     "url": "https://www.youtube.com/playlist?list=PL3D2QP2F5r9VDSj6YQb6Ihr_63Gxtm4L5", "lang": "mixed"},
    {"title": "Ein Buch Ein Tee", "url": "https://www.youtube.com/playlist?list=PLCospSPttrrVSk0N5Mqj1dveKZtDZNOAl",
     "lang": "ch"},
    {"title": "Ungerwegs Daheim", "url": "https://www.youtube.com/playlist?list=PLM4IdPP-Tx3W84w1GB8cn33GnuIGcqaeP",
     "lang": "ch"},
    {"title": "expectations - geplant und ungeplant kinderfrei",
     "url": "https://www.youtube.com/playlist?list=PL5ZbqYujTUkVmNCGMP4e0yFVhY8P5EC73", "lang": "ch"},
    {"title": "Wir müssen reden - Public Eye spricht Klartext",
     "url": "https://www.youtube.com/playlist?list=PLtTxFB6b5Pljl4RU6vimwfQpV490K6SQe", "lang": "de"},
    {"title": "FinanzFabio", "url": "https://www.youtube.com/playlist?list=PLGJjtm2tSyhQXU-_N2YkfqCffXhY6UHNe",
     "lang": "ch"},
    {"title": "Feel Good Podcast", "url": "https://www.youtube.com/playlist?list=PLf-k85Nq3_j-glR2im1SZv_BxqzdYdENk",
     "lang": "ch"},
    {"title": "Berner Jugendtreff", "url": "https://www.youtube.com/playlist?list=PLyWje_91744G6UAsfHjTLWDtejJdHmuYv",
     "lang": "ch"},
    {"title": "fadegrad", "url": "https://www.youtube.com/playlist?list=PL356t1Y2d_AXycvLzBF1n8ee0uM4pw9JX",
     "lang": "ch"},
    {"title": "Scho ghört", "url": "https://www.youtube.com/playlist?list=PLKaFe_fDMhQNbWvnJGC6HArb285ZUdGbz",
     "lang": "ch"},
    {"title": "Über den Bücherrand", "url": "https://www.youtube.com/playlist?list=PLPtjJ0sjI3yzhNtZUBY0_e462_gKtr90V",
     "lang": "mixed"},
    {"title": "Auf Bewährung: Leben mit Gefängnis",
     "url": "https://www.youtube.com/playlist?list=PLAD8a6PKLsRhHc-uS6fA6HTDijwE5Uwju", "lang": "mixed"},
    # {"title": "", "url": "", "lang": ""},
]


def download_yt_podcast_metadata(podcast: str, url: str, skip: bool = True) -> None:
    if not url:
        raise ValueError(f"Missing YouTube playlist URL for podcast '{podcast}'")

    saved_podcasts = get_downloaded_metadata()

    if skip and f"{podcast}.csv" in saved_podcasts:
        logger.warning(f"Podcast {podcast} already downloaded")
        return

    episodes = []
    pl = Playlist(url, 'WEB_EMBED')
    logger.info(f"Downloading metadata for podcast {podcast} with {pl.length} episodes.")
    for i, video_url in enumerate(pl.video_urls):
        try:
            try:
                _ = video_id(video_url)
            except RegexMatchError as e:
                logger.error(
                    "Skipping invalid YouTube URL for podcast '%s': %s. Error: %s",
                    podcast,
                    video_url,
                    str(e),
                )
                continue

            video = _create_youtube_with_fallback(video_url)
            episodes.append({
                "id": video.video_id,
                "title": video.title,
                "description": video.description if video.description else "NO_DESCRIPTION",
                "date_published": str(video.publish_date),
                "duration_s": video.length,
                "age_restricted": video.age_restricted,
                "url": video.watch_url
            })
        except VideoUnavailable as e:
            logger.error(
                "Video unavailable for podcast '%s': %s. Error: %s",
                podcast,
                video_url,
                str(e),
            )
            continue
        except RegexMatchError as e:
            logger.error(
                "Playlist parsing failed for podcast '%s'. URL: %s. Playlist: %s. Error: %s",
                podcast,
                video_url,
                url,
                str(e),
            )
            raise

    save_podcast_metadata_to_csv(podcast, episodes)
    logger.info(f"Expected number of podcasts: {pl.length}, saved {len(episodes)}")


def download_yt_podcast_audio(podcast: str) -> None:
    df = load_podcast_metadata_from_csv(podcast)
    create_audio_folder_if_not_exists(podcast)
    podcast_path = get_podcast_path(podcast)
    for i, metadata in df.iterrows():
        ep_path = os.path.join(podcast_path, metadata['id'] + ".mp3")
        if os.path.exists(ep_path):
            continue
        reset_cache()

        try:
            y = _create_youtube_with_fallback(metadata["url"])
            y.streams.get_audio_only().download(output_path=podcast_path, filename=f"{metadata['id']}.mp3",
                                                max_retries=4)
            time.sleep(0.15)
        except VideoUnavailable as e:
            logger.error(f"Did not download: {metadata['id']} for {podcast} because video was not available: {str(e)}")
            if 'url' in metadata and pd.notna(metadata['url']):
                logger.error(f"Video URL: {metadata['url']}")
            continue

        logger.info(f"Downloaded {metadata['title']} for {podcast}.")
    logger.info(f"Finished downloading {podcast}.")


def _create_youtube_with_fallback(url: str) -> YouTube:
    clients = ("ANDROID", "IOS", "WEB", "WEB_EMBED")
    last_exc: Exception | None = None

    for client in clients:
        try:
            return YouTube(url, client)
        except (VideoUnavailable, RegexMatchError) as e:
            last_exc = e
            continue

    if last_exc is not None:
        raise last_exc
    raise VideoUnavailable(f"Unable to create YouTube client for url: {url}")
