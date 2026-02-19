#!/usr/bin/env python3
"""
Video/Audio Processor - Extract, translate, review, and dub videos/audio
Supports: Local files (MP4, MP3, WAV, M4A, etc.) and URLs (YouTube, Twitter, etc.)
Usage: video_dubber.py <video_file_or_url> <target_lang> [groq_api_key]
"""
import sys
import os
import re
import subprocess
from pathlib import Path

# Import URL helper
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
from url_helper import is_url, download_from_url

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def get_video_info(video_file):
    """Get video duration and basic info"""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', video_file],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip())
    return {'duration': duration}

def transcribe_video(video_file, groq_api_key, source_lang='en'):
    """Transcribe video/audio using Groq Whisper Large V3"""
    from groq import Groq

    # Detect file type
    ext = Path(video_file).suffix.lower()
    if ext in ['.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac']:
        file_type = "Audio"
    else:
        file_type = "Video"

    print_header(f"🎵 Step 1: Transcribing {file_type}")
    print(f"Using: Groq Whisper Large V3")
    print(f"Language: {source_lang}")
    print(f"This should be very fast (20-30x realtime)...\n")

    client = Groq(api_key=groq_api_key)

    with open(video_file, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(video_file, audio_file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            language=source_lang,
            timestamp_granularities=["segment"]
        )

    # Convert to SRT
    srt_content = ""
    for i, segment in enumerate(transcription.segments, 1):
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()

        start_time = seconds_to_srt_time(start)
        end_time = seconds_to_srt_time(end)

        srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"

    # Save original SRT
    base_name = Path(video_file).stem
    original_srt = f"{base_name}_original.srt"
    with open(original_srt, 'w', encoding='utf-8') as f:
        f.write(srt_content)

    print(f"✅ Transcript saved: {original_srt}")
    print(f"   Segments: {len(transcription.segments)}")
    print(f"   Preview: {transcription.segments[0]['text'][:60]}...")

    return srt_content, original_srt

def seconds_to_srt_time(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def parse_srt(srt_content):
    """Parse SRT content into segments"""
    segments = []
    blocks = srt_content.strip().split('\n\n')

    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            index = lines[0]
            timestamp = lines[1]
            text = '\n'.join(lines[2:])

            match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})', timestamp)
            if match:
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match.groups())
                start_sec = h1*3600 + m1*60 + s1 + ms1/1000
                end_sec = h2*3600 + m2*60 + s2 + ms2/1000

                segments.append({
                    'index': int(index),
                    'timestamp': timestamp,
                    'start': start_sec,
                    'end': end_sec,
                    'text': text.strip()
                })

    return segments

def translate_subtitle(srt_content, target_lang, groq_api_key):
    """Translate SRT content to target language"""
    from groq import Groq

    print_header("🌐 Step 2: Translating Subtitles")
    print(f"Target language: {target_lang}")
    print(f"Using: Groq Llama 3.3 70B\n")

    client = Groq(api_key=groq_api_key)
    segments = parse_srt(srt_content)

    translated_segments = []
    for i, seg in enumerate(segments):
        print(f"  Translating segment {i+1}/{len(segments)}...", end='\r')

        # Build context-aware translation prompt
        translation_prompt = f"""Translate this video subtitle from English to {target_lang}.

CRITICAL RULES:
1. Use NATURAL {target_lang} phrasing - avoid word-for-word translation
2. Match the TONE and STYLE of the original (casual, formal, enthusiastic, etc.)
3. Keep technical terms, brand names, and proper nouns in ENGLISH (e.g., "Google Gemini", "ChatGPT", "YouTube")
4. Preserve the RHYTHM and FLOW for speech - translate for listening, not reading
5. Use colloquial expressions when appropriate - sound like a native speaker
6. Keep numbers, dates, and measurements in their original format

EXAMPLE (English to Chinese):
- Bad: "我想到了你，在我看到这个以后。" (machine translation)
- Good: "我一看到这个，就想到了你。" (natural Chinese)

SUBTITLE TEXT:
{seg['text']}

OUTPUT: Only the natural {target_lang} translation, nothing else."""

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional video subtitle translator specializing in natural, culturally-aware {target_lang} translations. Your translations sound like a native speaker, not a machine. You preserve the original tone, style, and intent while adapting idioms and expressions for the target culture."
                },
                {
                    "role": "user",
                    "content": translation_prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5  # Slightly higher for more natural phrasing
        )

        translated_text = response.choices[0].message.content.strip()
        translated_segments.append({
            'index': seg['index'],
            'timestamp': seg['timestamp'],
            'original': seg['text'],
            'translated': translated_text,
            'start': seg['start'],
            'end': seg['end']
        })

    print(f"\n✅ Translation complete!")
    return translated_segments

def display_translation_review(segments, max_display=5):
    """Display translation for review"""
    print_header("📝 Translation Review")
    print(f"Showing first {min(max_display, len(segments))} of {len(segments)} segments:\n")

    for i, seg in enumerate(segments[:max_display]):
        print(f"[{seg['index']}] {seg['timestamp']}")
        print(f"  EN: {seg['original']}")
        print(f"  →→: {seg['translated']}")
        print()

    if len(segments) > max_display:
        print(f"... and {len(segments) - max_display} more segments")
        print()

def save_translated_srt(segments, output_file):
    """Save translated segments to SRT file"""
    srt_content = ""
    for seg in segments:
        srt_content += f"{seg['index']}\n{seg['timestamp']}\n{seg['translated']}\n\n"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(srt_content)

    print(f"✅ Translated SRT saved: {output_file}")
    return srt_content

def main():
    if len(sys.argv) < 3:
        print("Usage: video_dubber.py <video_file_or_url> <target_lang> [groq_api_key]")
        print("Example: video_dubber.py video.mp4 chinese gsk_xxx")
        print("Example: video_dubber.py https://youtube.com/watch?v=xxx chinese gsk_xxx")
        print("Supports: Local files (MP4, MP3, WAV, M4A) and URLs (YouTube, Twitter, etc.)")
        sys.exit(1)

    input_source = sys.argv[1]
    target_lang = sys.argv[2]
    groq_api_key = sys.argv[3] if len(sys.argv) > 3 else os.getenv('GROQ_API_KEY')

    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY not provided")
        print("   Set via environment variable or pass as argument")
        sys.exit(1)

    # Check if input is URL or local file
    if is_url(input_source):
        # Download from URL
        video_file, file_type = download_from_url(input_source, output_dir=".")
        print(f"Downloaded {file_type}: {video_file}\n")
    else:
        # Local file
        video_file = input_source
        if not os.path.exists(video_file):
            print(f"❌ Error: File not found: {video_file}")
            sys.exit(1)

    base_name = Path(video_file).stem

    # Step 1: Transcribe
    original_srt_content, original_srt_file = transcribe_video(video_file, groq_api_key)

    # Step 2: Translate
    translated_segments = translate_subtitle(original_srt_content, target_lang, groq_api_key)

    # Step 3: Review (output for Claude to show user)
    display_translation_review(translated_segments)

    # Save translated SRT
    translated_srt_file = f"{base_name}_{target_lang}.srt"
    save_translated_srt(translated_segments, translated_srt_file)

    # Output status file for Claude to check
    status = {
        'video_file': video_file,
        'original_srt': original_srt_file,
        'translated_srt': translated_srt_file,
        'target_lang': target_lang,
        'segments': len(translated_segments),
        'status': 'awaiting_review'
    }

    import json
    with open(f"{base_name}_status.json", 'w') as f:
        json.dump(status, f, indent=2)

    print("\n" + "="*60)
    print("✅ Translation ready for review!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  • {original_srt_file} (original subtitles)")
    print(f"  • {translated_srt_file} (translated subtitles)")
    print(f"\nNext: Review translation and approve to continue to TTS generation")

if __name__ == "__main__":
    main()
