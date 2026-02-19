#!/usr/bin/env python3
"""
Generate comprehensive video/audio summary from transcript
Supports: Local files (MP4, MP3, WAV, M4A) and URLs (YouTube, Twitter, etc.)
Usage: video_summary.py <video_file_or_url> [target_lang] [groq_api_key]
"""
import sys
import os
from pathlib import Path

# Reuse existing transcription function
from video_dubber import transcribe_video, print_header
from url_helper import is_url, download_from_url

def generate_summary(transcript_text, target_lang, groq_api_key):
    """Generate comprehensive summary from transcript"""
    from groq import Groq

    print_header("📝 Step 2: Generating Summary")
    print(f"Language: {target_lang}")
    print(f"Using: Groq Llama 3.3 70B\n")

    client = Groq(api_key=groq_api_key)

    summary_prompt = f"""Analyze this video transcript and create a comprehensive summary.

TRANSCRIPT:
{transcript_text}

Generate a summary in {target_lang} with the following structure:

# Video Summary

## Overview
[2-3 sentence overview of what the video is about]

## Key Points
[5-7 bullet points of the most important takeaways]

## Detailed Summary
[2-3 paragraphs providing more context and details]

## Important Timestamps
[If specific moments are mentioned or important, list them with descriptions]

## Action Items
[If the video suggests actions to take, list them here. Otherwise omit this section]

RULES:
- Be concise but comprehensive
- Focus on the main message and key insights
- Preserve technical terms and proper nouns
- Use natural {target_lang} phrasing
- Include specific examples or statistics mentioned in the video

OUTPUT: Only the markdown-formatted summary, nothing else."""

    print("Analyzing transcript and generating summary...")

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are an expert video content analyst. You create clear, comprehensive summaries that capture the essence of video content. Your summaries are well-structured and easy to scan."
            },
            {
                "role": "user",
                "content": summary_prompt
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5
    )

    summary = response.choices[0].message.content.strip()
    return summary

def extract_plain_text_from_srt(srt_content):
    """Convert SRT to plain text transcript"""
    lines = []
    for block in srt_content.strip().split('\n\n'):
        block_lines = block.split('\n')
        if len(block_lines) >= 3:
            # Skip index and timestamp, take text
            text = '\n'.join(block_lines[2:])
            lines.append(text.strip())
    return '\n'.join(lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: video_summary.py <video_file_or_url> [target_lang] [groq_api_key]")
        print("Example: video_summary.py video.mp4")
        print("Example: video_summary.py video.mp4 chinese gsk_xxx")
        print("Example: video_summary.py https://youtube.com/watch?v=xxx chinese gsk_xxx")
        print("Supports: Local files (MP4, MP3, WAV, M4A) and URLs (YouTube, Twitter, etc.)")
        sys.exit(1)

    input_source = sys.argv[1]
    target_lang = sys.argv[2] if len(sys.argv) > 2 else "English"
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

    # Step 1: Transcribe video
    srt_content, srt_file = transcribe_video(video_file, groq_api_key, source_lang='en')

    # Convert SRT to plain text
    transcript_text = extract_plain_text_from_srt(srt_content)

    # Step 2: Generate summary
    summary = generate_summary(transcript_text, target_lang, groq_api_key)

    # Save summary
    summary_file = f"{base_name}_summary.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"\n✅ Summary saved: {summary_file}\n")

    # Show preview
    print("="*60)
    print("📄 Summary Preview")
    print("="*60)
    print()
    # Show first 800 characters
    preview = summary[:800]
    if len(summary) > 800:
        preview += "\n\n... (see full summary in file)"
    print(preview)
    print()
    print("="*60)
    print(f"Full summary: {summary_file}")
    print("="*60)

if __name__ == "__main__":
    main()
