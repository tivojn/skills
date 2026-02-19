#!/bin/bash
# Generate TTS audio with perfect subtitle sync and create dubbed video
# Usage: generate_tts_and_dub.sh <video_file> <original_srt> <translated_srt> <target_lang> [voice_profile] [voice_name]
#
# Uses numpy timeline assembly (scales to 1500+ segments).
# edge-tts runs async parallel (batches of 10) for speed.

set -e

VIDEO_FILE="$1"
ORIGINAL_SRT="$2"
TRANSLATED_SRT="$3"
TARGET_LANG="$4"
VOICE_PROFILE="$5"  # Optional: voicebox profile name OR "none" to skip
VOICE_NAME="$6"     # Optional: specific voice ID override (e.g. en-US-BrianNeural)

if [ -z "$VIDEO_FILE" ] || [ -z "$ORIGINAL_SRT" ] || [ -z "$TRANSLATED_SRT" ] || [ -z "$TARGET_LANG" ]; then
    echo "Usage: generate_tts_and_dub.sh <video_file> <original_srt> <translated_srt> <target_lang> [voice_profile] [voice_name]"
    echo "  voice_profile: voicebox profile name, or omit for auto-select"
    echo "  voice_name: specific voice ID (e.g. en-US-BrianNeural, am_michael)"
    exit 1
fi

BASE_NAME=$(basename "$VIDEO_FILE" | sed 's/\.[^.]*$//')
WORK_DIR="/tmp/tts_work_$$"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  Generating Synced TTS Audio"
echo "========================================"
echo ""
echo "Video: $VIDEO_FILE"
echo "Target: $TARGET_LANG"
echo ""

# Create working directory
mkdir -p "$WORK_DIR"

# Determine TTS engine
if [ -n "$VOICE_PROFILE" ] && [ "$VOICE_PROFILE" != "none" ]; then
    TTS_ENGINE="voicebox"
    echo "Using: Voicebox voice cloning (profile: $VOICE_PROFILE)"
elif [ "$TARGET_LANG" = "chinese" ] || [ "$TARGET_LANG" = "zh" ]; then
    if [ -f "$HOME/miniconda3/envs/kokoro/bin/python3" ]; then
        TTS_ENGINE="kokoro"
        echo "Using: Kokoro TTS (local)"
    else
        TTS_ENGINE="edge-tts"
        echo "Using: edge-tts (cloud, parallel)"
    fi
else
    TTS_ENGINE="edge-tts"
    echo "Using: edge-tts (cloud, parallel)"
fi
echo ""

# Build sync_tts.py arguments
SYNC_ARGS=("$TRANSLATED_SRT" "$WORK_DIR" "$TTS_ENGINE" "$TARGET_LANG")
if [ -n "$VOICE_PROFILE" ] && [ "$VOICE_PROFILE" != "none" ]; then
    SYNC_ARGS+=("$VOICE_PROFILE")
fi
if [ -n "$VOICE_NAME" ]; then
    # If no voice_profile but we have voice_name, add placeholder
    if [ -z "$VOICE_PROFILE" ] || [ "$VOICE_PROFILE" = "none" ]; then
        SYNC_ARGS+=("none")
    fi
    SYNC_ARGS+=("$VOICE_NAME")
fi

# Generate and sync TTS (parallel edge-tts + numpy timeline)
python3 "$SCRIPT_DIR/sync_tts.py" "${SYNC_ARGS[@]}"

echo ""

# The combined WAV is already built by sync_tts.py using numpy timeline
COMBINED_WAV="$WORK_DIR/combined.wav"

if [ ! -f "$COMBINED_WAV" ]; then
    echo "ERROR: Combined audio not found at $COMBINED_WAV"
    exit 1
fi

# Get video duration and trim/normalize
VIDEO_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$VIDEO_FILE")

echo "Trimming to video duration and normalizing..."
ffmpeg -y -i "$COMBINED_WAV" -t "$VIDEO_DUR" -af "volume=1.5" -ar 24000 -ac 1 "${BASE_NAME}_${TARGET_LANG}_audio.wav" 2>/dev/null

echo "Synced audio: ${BASE_NAME}_${TARGET_LANG}_audio.wav"
echo ""

# Create dubbed video with subtitle tracks
echo "========================================"
echo "  Creating Dubbed Video"
echo "========================================"
echo ""

# Try with dual subtitle tracks first
echo "Muxing audio + subtitles onto video..."
ffmpeg -y \
    -i "$VIDEO_FILE" \
    -i "${BASE_NAME}_${TARGET_LANG}_audio.wav" \
    -i "$ORIGINAL_SRT" \
    -i "$TRANSLATED_SRT" \
    -map 0:v:0 -map 1:a:0 -map 2:0 -map 3:0 \
    -c:v copy \
    -c:a aac -b:a 192k \
    -c:s mov_text \
    -metadata:s:s:0 language=eng -metadata:s:s:0 title="Original" \
    -metadata:s:s:1 language="${TARGET_LANG}" -metadata:s:s:1 title="${TARGET_LANG}" \
    -shortest \
    "${BASE_NAME}_dubbed.mp4" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Dual subs failed, trying with single subtitle track..."
    ffmpeg -y \
        -i "$VIDEO_FILE" \
        -i "${BASE_NAME}_${TARGET_LANG}_audio.wav" \
        -i "$TRANSLATED_SRT" \
        -map 0:v:0 -map 1:a:0 -map 2:0 \
        -c:v copy \
        -c:a aac -b:a 192k \
        -c:s mov_text -metadata:s:s:0 language="${TARGET_LANG}" \
        -shortest \
        "${BASE_NAME}_dubbed.mp4" 2>/dev/null

    if [ $? -ne 0 ]; then
        echo "Subtitle mux failed, creating video without subs..."
        ffmpeg -y \
            -i "$VIDEO_FILE" \
            -i "${BASE_NAME}_${TARGET_LANG}_audio.wav" \
            -map 0:v:0 -map 1:a:0 \
            -c:v copy \
            -c:a aac -b:a 192k \
            -shortest \
            "${BASE_NAME}_dubbed.mp4" 2>/dev/null
    fi
fi

echo ""

# Cleanup
echo "Cleaning up temporary files..."
rm -rf "$WORK_DIR"
rm -f "${BASE_NAME}_${TARGET_LANG}_audio.wav"

echo ""
echo "========================================"
echo "  DONE!"
echo "========================================"
echo ""
echo "Output files:"
echo "  - ${BASE_NAME}_original.srt"
echo "  - ${BASE_NAME}_${TARGET_LANG}.srt"
echo "  - ${BASE_NAME}_dubbed.mp4"
echo ""
echo "Video uses -c:v copy (no re-encode, fast)."
echo "Subtitle tracks embedded as soft subs (toggle in player)."
echo ""
