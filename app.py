from flask import Flask, request, render_template, redirect, url_for, jsonify, Response, stream_with_context
from flask_cors import CORS
#from pytubefix import YouTube
import imageio
from moviepy.editor import *
import moviepy.editor as mp
import os
import requests
import logging
from query import run_rag_query, save_chat_history
import subprocess
from fake_useragent import UserAgent
import yt_dlp
import random
import time
from threading import Semaphore, Lock
from urllib.parse import urlparse
import config
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timezone
import json
import uuid

# Set up logging
logging.basicConfig(
    level=getattr(logging, (config.LOG_LEVEL or "INFO").upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    force=True,
)
logger = logging.getLogger("youtube-rag")

app = Flask(__name__)
# Enable CORS for remote access (configurable via CORS_ORIGINS env var, default allows all)
CORS(app, origins=os.getenv('CORS_ORIGINS', '*').split(','), supports_credentials=True)

# Directory where downloaded MP3s will be saved
DOWNLOAD_FOLDER = config.DOWNLOAD_FOLDER
# Directory for text transcriptions
TEXT_FOLDER = config.TEXT_FOLDER
# API URL for transcription
API_URL = config.API_URL

# Ensure the download and text directories exist
def create_directories():
    """Create necessary directories for downloads and transcriptions."""
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(TEXT_FOLDER, exist_ok=True)

# Call the function to create directories
create_directories()

download_semaphore = Semaphore(3)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/56.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/11.1.2 Safari/605.1.15"
]

#
# Postgres table for duplicate detection / video registry
#
VideosBase = declarative_base()
videos_engine = create_engine(config.POSTGRES_CONNECTION_URI)
VideosSession = sessionmaker(bind=videos_engine)

class YouTubeVideo(VideosBase):
    __tablename__ = "youtube_videos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, unique=True, nullable=False, index=True)
    original_url = Column(Text, nullable=True)
    canonical_url = Column(Text, nullable=True)
    title = Column(Text, nullable=True)
    channel = Column(Text, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    mp3_path = Column(Text, nullable=True)
    transcript_path = Column(Text, nullable=True)
    added_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    ingested_at = Column(DateTime(timezone=True), nullable=True)

VideosBase.metadata.create_all(videos_engine)

# Progress tracking for SSE
_progress_store = {}
_progress_lock = Lock()

def _set_progress(task_id, stage, percent, message=""):
    """Update progress for a task."""
    with _progress_lock:
        _progress_store[task_id] = {
            "stage": stage,
            "percent": max(0, min(100, percent)),
            "message": message,
            "timestamp": time.time(),
        }

def _get_progress(task_id):
    """Get current progress for a task."""
    with _progress_lock:
        return _progress_store.get(task_id, {"stage": "unknown", "percent": 0, "message": ""})

def _clear_progress(task_id):
    """Clear progress after completion."""
    with _progress_lock:
        _progress_store.pop(task_id, None)

@contextmanager
def db_session():
    s = VideosSession()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'youtube-rag'})

@app.route('/progress/<task_id>', methods=['GET'])
def progress_stream(task_id):
    """SSE endpoint for progress updates."""
    def generate():
        last_percent = -1
        while True:
            progress = _get_progress(task_id)
            if progress["percent"] >= 100 or progress["stage"] == "error":
                yield f"data: {json.dumps(progress)}\n\n"
                break
            if progress["percent"] != last_percent:
                yield f"data: {json.dumps(progress)}\n\n"
                last_percent = progress["percent"]
            time.sleep(0.3)
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/run-ingest', methods=['POST'])
def run_ingest():
    try:
        data = request.get_json(silent=True) or {}
        task_id = data.get('task_id') or str(uuid.uuid4())
        logger.info("Received request to run ingest task_id=%s", task_id)

        # Run the inges.py script
        result = _run_ingest_script(task_id)

        logger.info("inges.py executed (rc=%s)", result['returncode'])
        logger.debug("inges.py stdout:\n%s", result['stdout'])
        logger.debug("inges.py stderr:\n%s", result['stderr'])

        # Return the output of the script
        result['task_id'] = task_id
        return jsonify(result)
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        return jsonify({'error': str(e)}), 500

def _run_ingest_script(task_id: str = None):
    """
    Internal helper to run the ingestion pipeline.
    Returns a dict shaped like the /run-ingest response.
    """
    if task_id:
        _set_progress(task_id, "ingesting", 0, "Starting ingestion...")
    t0 = time.monotonic()
    # Note: We can't easily track subprocess progress, so we estimate
    result = subprocess.run(['python', 'inges.py'], capture_output=True, text=True)
    dt_ms = int((time.monotonic() - t0) * 1000)
    if task_id:
        _set_progress(task_id, "complete" if result.returncode == 0 else "error", 100, 
                     f"Ingest complete (rc={result.returncode})")
    logger.info("Auto-ingest completed (rc=%s, %sms)", result.returncode, dt_ms)
    return {
        'stdout': result.stdout,
        'stderr': result.stderr,
        'returncode': result.returncode
    }

def _extract_youtube_info(youtube_url: str):
    """
    Extracts YouTube metadata (including stable video_id) without downloading.
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'restrictfilenames': True,
        'nocheckcertificate': True,
        'user-agent': random.choice(USER_AGENTS),
        'skip_download': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        logger.debug("Extracting yt metadata (no download) url=%s", youtube_url)
        info = ydl.extract_info(youtube_url, download=False)
    return {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("uploader") or info.get("channel"),
        "duration_seconds": info.get("duration"),
        "canonical_url": info.get("webpage_url") or youtube_url,
    }

def _download_transcribe_flow(youtube_url: str, task_id: str = None):
    """
    Download YouTube audio and transcribe it to a .txt file.
    Returns (mp3_file_path, transcript_file_path).
    """
    if not is_valid_youtube_url(youtube_url):
        raise ValueError("Invalid YouTube URL.")

    if task_id:
        _set_progress(task_id, "extracting", 5, "Extracting video metadata...")
    t0 = time.monotonic()
    info = _extract_youtube_info(youtube_url)
    video_id = info.get("video_id")
    if not video_id:
        raise RuntimeError("Could not determine YouTube video_id from URL.")

    if task_id:
        _set_progress(task_id, "checking", 10, "Checking for duplicates...")
    with db_session() as s:
        existing = s.query(YouTubeVideo).filter_by(video_id=video_id).first()
        if existing:
            logger.info("Duplicate video blocked video_id=%s url=%s", video_id, youtube_url)
            if task_id:
                _set_progress(task_id, "error", 0, f"Duplicate video: {video_id}")
            raise ValueError(
                f"Duplicate video: {video_id} already added. "
                f"Transcript: {existing.transcript_path or 'unknown'}"
            )

    if task_id:
        _set_progress(task_id, "downloading", 20, f"Downloading audio: {info.get('title', video_id)}...")
    logger.info("Processing youtube url=%s video_id=%s title=%s", youtube_url, video_id, info.get("title"))
    mp3_file_path = download_youtube_audio(youtube_url, DOWNLOAD_FOLDER)
    if not mp3_file_path:
        if task_id:
            _set_progress(task_id, "error", 0, "Download failed")
        raise RuntimeError("Failed to download and convert YouTube video to MP3.")

    if task_id:
        _set_progress(task_id, "transcribing", 60, "Transcribing audio...")
    transcript_file_path = transcribe_audio(mp3_file_path, config.API_URL, config.TEXT_FOLDER)
    if not transcript_file_path:
        if task_id:
            _set_progress(task_id, "error", 0, "Transcription failed")
        raise RuntimeError("Transcription failed.")

    if task_id:
        _set_progress(task_id, "registering", 90, "Registering video...")
    # Register the video (for duplicate detection)
    try:
        with db_session() as s:
            s.add(YouTubeVideo(
                video_id=video_id,
                original_url=youtube_url,
                canonical_url=info.get("canonical_url"),
                title=info.get("title"),
                channel=info.get("channel"),
                duration_seconds=info.get("duration_seconds"),
                mp3_path=mp3_file_path,
                transcript_path=transcript_file_path,
            ))
    except IntegrityError:
        # Race condition: another worker added it after our pre-check
        if task_id:
            _set_progress(task_id, "error", 0, f"Duplicate video: {video_id}")
        raise ValueError(f"Duplicate video: {video_id} already added.")

    if task_id:
        _set_progress(task_id, "complete", 100, "Download and transcribe complete!")
    dt_ms = int((time.monotonic() - t0) * 1000)
    logger.info("Completed download+transcribe video_id=%s (%sms) mp3=%s txt=%s", video_id, dt_ms, mp3_file_path, transcript_file_path)
    return mp3_file_path, transcript_file_path, info

@app.route('/download-transcribe', methods=['POST'])
def download_transcribe():
    """
    JSON endpoint for the web UI to download/transcribe without redirecting.
    Body:
      { "youtube_url": "...", "ingest": true|false, "task_id": "..." }
    """
    try:
        t0 = time.monotonic()
        data = request.get_json(silent=True) or {}
        youtube_url = (data.get('youtube_url') or '').strip()
        ingest = bool(data.get('ingest', False))
        task_id = data.get('task_id') or str(uuid.uuid4())
        logger.debug("POST /download-transcribe ingest=%s url=%s task_id=%s", ingest, youtube_url, task_id)

        if not youtube_url:
            return jsonify({'error': 'youtube_url is required'}), 400

        mp3_file_path, transcript_file_path, info = _download_transcribe_flow(youtube_url, task_id)

        response = {
            'mp3_file': mp3_file_path,
            'transcript_file': transcript_file_path,
            'video_id': info.get("video_id"),
            'title': info.get("title"),
        }

        if ingest:
            logger.info("Auto-ingest requested after transcription")
            response['ingest'] = _run_ingest_script(task_id)
            if response['ingest'].get("returncode") == 0 and response.get("video_id"):
                try:
                    with db_session() as s:
                        row = s.query(YouTubeVideo).filter_by(video_id=response["video_id"]).first()
                        if row:
                            row.ingested_at = datetime.now(timezone.utc)
                except Exception:
                    logger.exception("Failed to update ingested_at")

        dt_ms = int((time.monotonic() - t0) * 1000)
        logger.info("POST /download-transcribe ok video_id=%s (%sms)", response.get("video_id"), dt_ms)
        response['task_id'] = task_id
        return jsonify(response)
    except ValueError as e:
        logger.warning(str(e))
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("POST /download-transcribe failed: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def run_query():
    try:
        data = request.get_json()
        query = data.get('query')
        user_id = data.get('user_id')  # optional user_id
        task_id = data.get('task_id') or str(uuid.uuid4())

        logger.info("Query received user_id=%s q_len=%s task_id=%s", user_id, len(query or ""), task_id)
        logger.debug("Query text: %s", query)

        if task_id:
            _set_progress(task_id, "searching", 20, "Querying remote LightRAG...")
        if task_id:
            _set_progress(task_id, "generating", 60, "Generating response...")
        response = run_rag_query(query or "")
        if not response or not str(response).strip():
            if task_id:
                _set_progress(task_id, "error", 0, "No answer produced")
            return jsonify({
                "response": "No answer produced. Run ingest on your transcripts (remote LightRAG), then try again.",
                "task_id": task_id
            })

        if task_id:
            _set_progress(task_id, "complete", 100, "Response generated")
        logger.info("Generated response len=%s", len(response or ""))
        logger.debug("Generated response: %s", response)

        # Save chat history
        save_chat_history(user_id, query, response)

        return jsonify({"response": response, "task_id": task_id})
    except Exception as e:
        if 'task_id' in locals():
            _set_progress(task_id, "error", 0, str(e))
        logger.exception("POST /query failed: %s", str(e))
        return jsonify({'error': str(e), "task_id": task_id if 'task_id' in locals() else None}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        youtube_url = request.form.get('youtube_url')
        if youtube_url:
            try:
                _download_transcribe_flow(youtube_url)
                return redirect(url_for('success'))
            except ValueError:
                logging.warning("Invalid YouTube URL provided.")
                return "Invalid YouTube URL."
            except Exception:
                logging.exception("Failed to download/transcribe.")
                return "Failed to download and/or transcribe YouTube video."
        else:
            logging.warning("No URL provided.")
            return "No URL provided."
    return render_template('index.html')

@app.route('/success')
def success():
    return "The audio has been successfully downloaded and transcribed!"

def is_valid_youtube_url(url):
    parsed = urlparse(url)
    if parsed.hostname in ['www.youtube.com', 'youtube.com', 'youtu.be']:
        return True
    return False

def download_youtube_audio(youtube_url, output_dir):
    temp_file_path = None
    try:
        with download_semaphore:
            # Random delay between 1 to 3 seconds to mimic human behavior
            delay = random.uniform(1, 3)
            logging.info(f"Sleeping for {delay:.2f} seconds to throttle requests.")
            time.sleep(delay)

            # Configure yt_dlp options without proxies
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'user-agent': random.choice(USER_AGENTS),
                'quiet': True,
                'no_warnings': True,
                'restrictfilenames': True,
                # To avoid downloading metadata, set to download audio only
                'nocheckcertificate': True,
            }
            logging.info(f"Using User-Agent: {ydl_opts['user-agent']}")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info("Starting yt-dlp download url=%s", youtube_url)
                info_dict = ydl.extract_info(youtube_url, download=True)
                video_id = info_dict.get("id", None)
                ext = info_dict.get("ext", None)
                temp_file_path = os.path.join(output_dir, f"{video_id}.{ext}")

                mp3_output_path = os.path.join(output_dir, f"{video_id}.mp3")
                if os.path.exists(mp3_output_path):
                    logger.info("Downloaded and converted to MP3: %s", mp3_output_path)
                    return mp3_output_path
                else:
                    logger.error("MP3 file was not created.")
                    return None

    except yt_dlp.utils.DownloadError as e:
        logging.error(f"Download error: {e}")
        return None
    except yt_dlp.utils.ExtractorError as e:
        logging.error(f"Extractor error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Transcription request failed: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None
    finally:
        # No need to manually remove temporary files as yt_dlp handles it
        pass

def transcribe_audio(mp3_file_path, api_url, text_output_dir):
    try:
        with open(mp3_file_path, "rb") as audio_file:
            files = {'file': audio_file}
            logger.info("Sending audio for transcription: %s -> %s", mp3_file_path, api_url)
            response = requests.post(api_url, files=files)
        if response.status_code == 200:
            transcript = response.json()
            if 'text' in transcript:
                logger.info("Transcription successful.")
                # Ensure the text output directory exists
                if not os.path.exists(text_output_dir):
                    os.makedirs(text_output_dir)
                transcript_file_name = os.path.basename(mp3_file_path).replace('.mp3', '.txt')
                transcript_file_path = os.path.join(text_output_dir, transcript_file_name)
                with open(transcript_file_path, 'w', encoding='utf-8') as f:
                    f.write(transcript['text'])
                logger.debug("Transcript written: %s (chars=%s)", transcript_file_path, len(transcript.get('text') or ""))
                return transcript_file_path
            else:
                logger.error("Error: 'text' key not found in the response.")
                return None
        else:
            logger.error("Transcription error HTTP %s body=%s", response.status_code, response.text[:500])
            return None
    except requests.exceptions.RequestException as e:
        logger.error("Transcription request failed: %s", e)
        return None
    except FileNotFoundError:
        logger.error("File not found: %s", mp3_file_path)
        return None

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5004))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('DEBUG', 'false').lower() == 'true')
