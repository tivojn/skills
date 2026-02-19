#!/usr/bin/env python3
"""
URL Helper - Download video/audio from URLs
Supports: YouTube, Twitter, TikTok, Instagram, and many more via yt-dlp
"""
import subprocess
import sys
import re
from pathlib import Path
from urllib.parse import urlparse

def is_url(input_string):
    """Check if string is a URL"""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(input_string) is not None

def download_from_url(url, output_dir="."):
    """
    Download video/audio from URL using yt-dlp
    Returns: (downloaded_file_path, file_type)
    """
    print(f"\n🌐 Downloading from URL...")
    print(f"URL: {url}")
    print(f"Using: yt-dlp\n")

    # Generate output filename template
    output_template = f"{output_dir}/%(title)s.%(ext)s"

    # Download with best quality, prefer mp4/m4a
    cmd = [
        'yt-dlp',
        '--no-playlist',  # Don't download playlists
        '--format', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        '--merge-output-format', 'mp4',
        '--output', output_template,
        '--print', 'after_move:filepath',  # Print final filepath
        url
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Extract the downloaded file path from output
        # yt-dlp prints the filepath on the last line
        lines = result.stdout.strip().split('\n')
        downloaded_file = None

        for line in reversed(lines):
            if line and Path(line).exists():
                downloaded_file = line
                break

        if not downloaded_file:
            raise Exception("Could not determine downloaded file path")

        print(f"✅ Downloaded: {downloaded_file}\n")

        # Determine file type
        ext = Path(downloaded_file).suffix.lower()
        if ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv']:
            file_type = 'video'
        elif ext in ['.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac']:
            file_type = 'audio'
        else:
            file_type = 'unknown'

        return downloaded_file, file_type

    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading from URL:")
        print(e.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def download_audio_only(url, output_dir="."):
    """
    Download only audio from URL (useful for audio processing)
    Returns: downloaded_audio_file_path
    """
    print(f"\n🌐 Downloading audio from URL...")
    print(f"URL: {url}")
    print(f"Using: yt-dlp (audio only)\n")

    output_template = f"{output_dir}/%(title)s.%(ext)s"

    cmd = [
        'yt-dlp',
        '--no-playlist',
        '--extract-audio',
        '--audio-format', 'mp3',
        '--audio-quality', '0',  # Best quality
        '--output', output_template,
        '--print', 'after_move:filepath',
        url
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        lines = result.stdout.strip().split('\n')
        downloaded_file = None

        for line in reversed(lines):
            if line and Path(line).exists():
                downloaded_file = line
                break

        if not downloaded_file:
            raise Exception("Could not determine downloaded file path")

        print(f"✅ Downloaded: {downloaded_file}\n")
        return downloaded_file

    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading audio:")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: url_helper.py <url> [output_dir]")
        print("Example: url_helper.py https://youtube.com/watch?v=xxx")
        sys.exit(1)

    url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    if not is_url(url):
        print(f"❌ Error: Not a valid URL: {url}")
        sys.exit(1)

    downloaded_file, file_type = download_from_url(url, output_dir)
    print(f"File type: {file_type}")
    print(f"Path: {downloaded_file}")
