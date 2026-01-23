import base64
import json
import os
import time
from urllib.parse import parse_qs, urlparse

import requests

from src.download.utils import save_podcast_metadata_to_csv, load_podcast_metadata_from_csv, \
    create_audio_folder_if_not_exists, PODCAST_METADATA_FOLDER, PODCAST_AUDIO_FOLDER, get_downloaded_metadata
from src.utils.logger import get_logger

CONSUMER_KEY = os.getenv("SRF_CONSUMER_KEY", "YOUR_CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("SRF_CONSUMER_SECRET", "YOUR_CONSUMER_SECRET")
AUTH_TOKEN = base64.b64encode(f"{CONSUMER_KEY}:{CONSUMER_SECRET}".encode()).decode()

URL_BASE = "https://api.srgssr.ch"
URL_CLIENT_CREDENTIALS = f"{URL_BASE}/oauth/v1/accesstoken?grant_type=client_credentials"
URL_AUDIOS = f"{URL_BASE}/audiometadata/v2"

logger = get_logger(__name__)
_MISSING_KEY_LOGGED: set[tuple[str, str]] = set()

new_podcasts = [
    "Besserwisser",
    "Blick in die Feuilletons",
    "BuchZeichen",
    "Die verflixte Gebrauchsanweisung",
    "Echo der Zeit",
    "Einfach Politik",
    "Es geschah am... Postraub des Jahrhunderts",
    "Espresso",
    "Focus",
    "Forum",
    "Grauen",
    "Input",
    "Kontext",
    "Krimi",
    "Kultur kompakt",
    "News Plus",
    "Persönlich",
    "Perspektiven",
    "Politikum",
    "Ratgeber",
    "Rehmann",
    "Trend",
    "Trüffelschweine",
    "Wetter",
    "WortSchatz",
    "Zeitblende",
    "Schweizerdeutsch hat keine Zukunft!"
]

existing_podcasts = [  # 17278872.0 seconds
    "Debriefing 404",
    "Digital Podcast",
    "Dini Mundart",
    # "Dini Mundart Schnabelweid",  # <- no downloads available
    "Gast am Mittag",
    "Geek-Sofa",
    # "Giigets Die SRF 3-Alltagsphilosophie", # <- no downloads available
    # "Morgengast",  # <- no downloads available
    "Pipifax",
    "Podcast am Pistenrand",
    "Samstagsrundschau",
    # "Schwiiz und dütlich",  # <-- no downloads available
    "#SRFglobal",
    "Sykora Gisler",
    "Tagesgespräch",
    "Ufwärmrundi",
    "Vetters Töne",
    "Wetterfrage",
    "Zivadiliring",
    "Zytlupe",
    ############# from here its de only #####################
    "100 Sekunden Wissen",
    # "Kontext",  # can contain background noise
    "Kultur-Talk",
    "Kopf voran",
    "Literaturclub: Zwei mit Buch",
    "Medientalk",
    "Sternstunde Philosophie",
    "Sternstunde Religion",
    "Wirtschaftswoche",
    "Wissenschaftsmagazin"  # mixed (EN, DE, CHDE), can contain background noise
]


def _check_and_load_response(response: requests.Response) -> dict:
    if response.status_code in [200, 203]:
        return json.loads(response.text)
    else:
        raise RuntimeError(f"Failed to get response. Response code {response.status_code} with message {response.text}")


def get_access_token() -> dict:
    headers = {
        "Authorization": "Basic " + AUTH_TOKEN,
        "Cache-Control": "no-cache",
        "Content-Length": "0",
    }
    response = requests.post(URL_CLIENT_CREDENTIALS, headers=headers)
    return _check_and_load_response(response)


def _get_key(payload: dict, key: str, context: str):
    try:
        return payload[key]
    except KeyError:
        logger.error(
            "Missing '%s' in %s. Available keys: %s",
            key,
            context,
            sorted(payload.keys()),
        )
        logger.error("%s payload: %s", context, json.dumps(payload, ensure_ascii=True))
        raise


def _log_missing_key_once(context: str, key: str, payload: dict) -> None:
    cache_key = (context, key)
    if cache_key in _MISSING_KEY_LOGGED:
        return
    _MISSING_KEY_LOGGED.add(cache_key)
    logger.warning(
        "Missing '%s' in %s. Available keys: %s",
        key,
        context,
        sorted(payload.keys()),
    )
    logger.warning("%s payload: %s", context, json.dumps(payload, ensure_ascii=True))


def _parse_next_token(next_value: str) -> str | None:
    if next_value.isdigit():
        return next_value
    parsed = urlparse(next_value)
    query = parsed.query or next_value
    params = parse_qs(query)
    return params.get("next", [None])[0]


def _normalize_title(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _collect_metadata(media: list, current_podcast) -> list:
    episodes = []
    current_norm = _normalize_title(current_podcast)
    seen_titles: set[str] = set()
    for episode in media:
        show = _get_key(episode, "show", "episode")
        show_title = _get_key(show, "title", "episode.show")
        seen_titles.add(show_title)
        show_norm = _normalize_title(show_title)
        if current_norm == show_norm or current_norm in show_norm or show_norm in current_norm:
            try:
                duration = episode["duration"]
            except KeyError:
                logger.error("Missing 'duration' in episode. Keys: %s", sorted(episode.keys()))
                logger.error("episode payload: %s", json.dumps(episode, ensure_ascii=True))
                raise

            download_available = episode.get("downloadAvailable", False)
            if "downloadAvailable" not in episode:
                _log_missing_key_once("episode", "downloadAvailable", episode)
            subtitles_available = episode.get("subtitlesAvailable", False)
            if "subtitlesAvailable" not in episode:
                _log_missing_key_once("episode", "subtitlesAvailable", episode)
            url = episode.get("podcastHdUrl") or episode.get("podcastSdUrl") or "NO_URL"
            if url == "NO_URL":
                download_available = False
            else:
                download_available = True

            episodes.append({
                "id": _get_key(episode, "id", "episode"),
                "title": _get_key(episode, "title", "episode"),
                "description": episode.get("description", "NO_DESCRIPTION"),
                "date_published": episode.get("date", "NO_DATE"),
                "duration_s": duration / 1000,
                "download_available": download_available,
                "subtitles_available": subtitles_available,
                "url": url,
            })
    if not episodes and media:
        sample_titles = sorted(seen_titles)[:10]
        logger.warning(
            "No episode title matches for '%s'. Sample show titles: %s",
            current_podcast,
            sample_titles,
        )
    return episodes


def process_srf_podcast(podcast: str) -> None:
    download_srf_podcast_metadata(podcast)
    download_srf_podcast_audio(podcast)


def download_srf_podcast_metadata(podcast: str, skip: bool = True) -> None:
    if not podcast or not podcast.strip():
        raise ValueError("podcast name is empty; set 'podcast_name' in config.yaml")

    def _get_media_list(payload: dict) -> list:
        try:
            return payload["searchResultMediaList"]
        except KeyError:
            logger.error(
                "Missing 'searchResultMediaList' in SRF response. Available keys: %s",
                sorted(payload.keys()),
            )
            logger.error("SRF response payload: %s", json.dumps(payload, ensure_ascii=True))
            raise

    os.makedirs(PODCAST_METADATA_FOLDER, exist_ok=True)
    url = URL_AUDIOS + "/audios/search"

    access_token = get_access_token()
    headers = {
        "Authorization": f"{access_token['token_type']} {access_token['access_token']}",
        "Cache-Control": "no-cache",
        "accept": "application/json"
    }

    saved_podcasts = get_downloaded_metadata()

    if skip and f"{podcast}.csv" in saved_podcasts:
        logger.warning(f"Podcast {podcast} already downloaded")
        return

    params = {
        "bu": "srf",
        "q": podcast,
        "pageSize": 100
    }

    response = requests.get(url, headers=headers, params=params)
    json_response = _check_and_load_response(response)

    episodes = []
    total_episodes = _get_key(json_response, "total", "response")
    logger.info(f"Getting podcast {podcast} with total number of episodes: {total_episodes}")
    episodes.extend(_collect_metadata(_get_media_list(json_response), podcast))

    while "next" in json_response:
        next_token = _parse_next_token(str(json_response["next"]))
        if not next_token:
            logger.error(
                "Could not parse next token for %s. next=%s",
                podcast,
                json_response.get("next"),
            )
            break
        params["next"] = next_token
        try:
            response = requests.get(url, headers=headers, params=params)
            json_response = _check_and_load_response(response)
        except Exception as e:
            logger.error("Pagination request failed for %s: %s", podcast, str(e))
            break
        episodes.extend(_collect_metadata(_get_media_list(json_response), podcast))

    if not episodes:
        logger.error(
            "No episodes collected for %s. Expected %s episodes from SRF response.",
            podcast,
            total_episodes,
        )
        raise RuntimeError("No episodes collected from SRF API response.")

    downloadable = sum(
        1 for episode in episodes if episode.get("download_available") and episode.get("url") != "NO_URL"
    )
    save_podcast_metadata_to_csv(podcast, episodes)
    logger.info(
        "Expected episodes: %s, saved %s, downloadable %s",
        total_episodes,
        len(episodes),
        downloadable,
    )


def download_srf_podcast_audio(podcast: str) -> None:
    if not podcast or not podcast.strip():
        raise ValueError("podcast name is empty; set 'podcast_name' in config.yaml")
    df = load_podcast_metadata_from_csv(podcast)
    create_audio_folder_if_not_exists(podcast)

    for i, metadata in df.iterrows():
        ep_path = f"{PODCAST_AUDIO_FOLDER}/{podcast}/{metadata['id']}.mp3"
        if not metadata["download_available"] or os.path.exists(ep_path):
            continue

        response = requests.get(metadata["url"], allow_redirects=True)
        with open(ep_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"downloaded {metadata['id']} for {podcast}")
        time.sleep(0.25)
