#!/usr/bin/env python3
"""
Sync TTS audio to subtitle timing with parallel generation and numpy timeline assembly.
Usage: sync_tts.py <translated_srt> <work_dir> <tts_engine> <target_lang> [voice_profile] [voice_name]

Supports:
  - edge-tts: Async parallel generation (batches of 10), 50+ languages
  - kokoro: Local TTS via Kokoro-82M, English/Chinese/Japanese/etc.
  - voicebox: Voice cloning via mlx-audio Qwen3-TTS

Timeline assembly uses numpy array placement (scales to 1500+ segments).
"""
import sys
import os
import json
import re
import subprocess
import time
import asyncio
import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000


def parse_srt(srt_file):
    """Parse SRT file and return segments with timing"""
    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    segments = []
    for i, block in enumerate(content.strip().split('\n\n')):
        lines = block.split('\n')
        if len(lines) >= 3:
            text = '\n'.join(lines[2:]).strip()
            m = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
            if m:
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
                start = h1*3600 + m1*60 + s1 + ms1/1000
                end = h2*3600 + m2*60 + s2 + ms2/1000
                duration = end - start
                segments.append({
                    'index': i,
                    'text': text,
                    'start': start,
                    'end': end,
                    'duration': duration
                })
    return segments


# ============================================================
# TTS Generation
# ============================================================

async def generate_edge_tts_all(segments, work_dir, voice):
    """Generate all segments with edge-tts in parallel batches"""
    import edge_tts

    BATCH_SIZE = 10
    total = len(segments)
    t0 = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        tasks = []

        for i in range(batch_start, batch_end):
            text = segments[i]['text'].strip()
            if not text:
                text = "..."
            out_path = os.path.join(work_dir, f"raw_{segments[i]['index']:04d}.mp3")

            if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
                continue  # skip already generated

            async def gen_one(idx, txt, path):
                try:
                    communicate = edge_tts.Communicate(txt, voice)
                    await communicate.save(path)
                except Exception as e:
                    try:
                        await asyncio.sleep(1)
                        communicate = edge_tts.Communicate(txt, voice)
                        await communicate.save(path)
                    except Exception:
                        print(f"  FAIL {idx+1}: {e}")

            tasks.append(gen_one(segments[i]['index'], text, out_path))

        if tasks:
            await asyncio.gather(*tasks)

        elapsed = time.time() - t0
        pct = batch_end / total * 100
        eta = elapsed / max(batch_end, 1) * (total - batch_end)
        print(f"  TTS: {batch_end}/{total} ({pct:.0f}%) - elapsed {elapsed:.0f}s - ETA {eta:.0f}s")


def generate_edge_tts(segments, work_dir, voice):
    """Wrapper to run async edge-tts generation"""
    asyncio.run(generate_edge_tts_all(segments, work_dir, voice))


def generate_kokoro_tts(segments, work_dir, voice='am_michael'):
    """Generate all segments with Kokoro TTS in a single process"""
    kokoro_py = os.path.expanduser("~/miniconda3/envs/kokoro/bin/python3")
    total = len(segments)

    # Build segment data for the kokoro script
    seg_data = []
    for seg in segments:
        seg_data.append({
            'index': seg['index'],
            'text': seg['text'].strip() or '...',
            'out_path': os.path.join(work_dir, f"raw_{seg['index']:04d}.wav")
        })

    # Determine lang_code from voice name
    lang_code = 'a'  # American English default
    if voice.startswith('z'):
        lang_code = 'z'  # Chinese

    seg_json = json.dumps(seg_data)
    gen_script = f'''
import time, json, os, warnings
warnings.filterwarnings("ignore")
start_all = time.time()

from kokoro import KPipeline
import numpy as np
import soundfile as sf

pipe = KPipeline(lang_code="{lang_code}", repo_id="hexgrad/Kokoro-82M")
segs = json.loads({repr(seg_json)})
VOICE = "{voice}"

for i, seg in enumerate(segs):
    if os.path.exists(seg["out_path"]) and os.path.getsize(seg["out_path"]) > 100:
        continue
    t0 = time.time()
    text = seg["text"]
    generator = pipe(text, voice=VOICE, speed=1.0)
    audio_chunks = []
    for gs, ps, audio in generator:
        audio_chunks.append(audio)
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        sf.write(seg["out_path"], full_audio, 24000)
    else:
        audio = np.zeros(2400, dtype=np.float32)
        sf.write(seg["out_path"], audio, 24000)
    if (i+1) % 50 == 0 or i == len(segs)-1:
        elapsed = time.time() - start_all
        print(f"  Kokoro: {{i+1}}/{{len(segs)}} - {{elapsed:.0f}}s")

print(f"Total Kokoro generation: {{time.time()-start_all:.1f}}s")
'''

    result = subprocess.run(
        [kokoro_py, '-c', gen_script],
        capture_output=True, text=True, timeout=600
    )
    print(result.stdout)
    if result.returncode != 0:
        errors = [l for l in result.stderr.split('\n') if 'Error' in l or 'error' in l.lower()]
        if errors:
            print("Errors:", '\n'.join(errors[:5]))


def generate_voicebox_tts(segments, work_dir, voice_profile):
    """Generate all segments with voicebox voice cloning"""
    voicebox_script = os.path.expanduser("~/.claude/skills/voicebox/scripts/voicebox.py")
    total = len(segments)

    for i, seg in enumerate(segments):
        out_path = os.path.join(work_dir, f"raw_{seg['index']:04d}.wav")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            continue

        text = seg['text'].strip()
        if not text:
            silence = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
            sf.write(out_path, silence, SAMPLE_RATE)
            continue

        subprocess.run(
            ['uv', 'run', voicebox_script, 'generate', voice_profile, text, '--quality', 'high'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        tmp_out = '/tmp/voicebox_output.wav'
        if os.path.exists(tmp_out):
            import shutil
            shutil.copy(tmp_out, out_path)

        if (i+1) % 10 == 0 or i == total-1:
            print(f"  Voicebox: {i+1}/{total}")


# ============================================================
# Speed Adjustment
# ============================================================

def speed_adjust_all(segments, work_dir):
    """Speed-adjust all segments to match SRT duration"""
    total = len(segments)
    t0 = time.time()

    for i, seg in enumerate(segments):
        target_dur = seg['duration']
        idx = seg['index']

        # Find raw file (mp3 for edge-tts, wav for kokoro/voicebox)
        raw_mp3 = os.path.join(work_dir, f"raw_{idx:04d}.mp3")
        raw_wav = os.path.join(work_dir, f"raw_{idx:04d}.wav")
        adjusted_path = os.path.join(work_dir, f"adj_{idx:04d}.wav")

        if os.path.exists(adjusted_path) and os.path.getsize(adjusted_path) > 100:
            continue

        # Determine input file
        if os.path.exists(raw_mp3) and os.path.getsize(raw_mp3) > 100:
            input_file = raw_mp3
        elif os.path.exists(raw_wav) and os.path.getsize(raw_wav) > 100:
            input_file = raw_wav
        else:
            # Create silence for missing segments
            silence = np.zeros(max(int(target_dur * SAMPLE_RATE), SAMPLE_RATE // 10), dtype=np.float32)
            sf.write(adjusted_path, silence, SAMPLE_RATE)
            continue

        # Get actual duration
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', input_file],
            capture_output=True, text=True
        )
        try:
            actual_dur = float(result.stdout.strip())
        except (ValueError, AttributeError):
            silence = np.zeros(max(int(target_dur * SAMPLE_RATE), SAMPLE_RATE // 10), dtype=np.float32)
            sf.write(adjusted_path, silence, SAMPLE_RATE)
            continue

        if target_dur <= 0.05:
            # Very short segment, just convert without speed adjustment
            subprocess.run(
                ['ffmpeg', '-y', '-i', input_file, '-ar', str(SAMPLE_RATE), '-ac', '1', adjusted_path],
                capture_output=True, text=True
            )
        else:
            ratio = actual_dur / target_dur
            if ratio < 0.5:
                ratio = 0.5
            elif ratio > 4.0:
                ratio = 4.0

            # Build atempo filter chain (each atempo supports 0.5-2.0)
            filters = []
            r = ratio
            while r > 2.0:
                filters.append("atempo=2.0")
                r /= 2.0
            while r < 0.5:
                filters.append("atempo=0.5")
                r *= 2.0
            filters.append(f"atempo={r:.6f}")
            filter_str = ",".join(filters)

            subprocess.run([
                'ffmpeg', '-y', '-i', input_file,
                '-filter:a', filter_str,
                '-ar', str(SAMPLE_RATE), '-ac', '1',
                adjusted_path
            ], capture_output=True, text=True)

        if (i+1) % 100 == 0 or i == total-1:
            elapsed = time.time() - t0
            print(f"  Adjusted: {i+1}/{total} ({(i+1)/total*100:.0f}%) - {elapsed:.0f}s")


# ============================================================
# Numpy Timeline Assembly
# ============================================================

def build_numpy_timeline(segments, work_dir, output_audio):
    """Build full audio timeline using numpy array placement.

    This approach scales to 1500+ segments without hitting ffmpeg input limits.
    Each adjusted WAV is placed at its exact SRT start position in a pre-allocated array.
    """
    total = len(segments)

    # Total duration from last segment end + 2s buffer
    total_dur = segments[-1]['end'] + 2.0
    total_samples = int(total_dur * SAMPLE_RATE)
    timeline = np.zeros(total_samples, dtype=np.float32)

    for i, seg in enumerate(segments):
        idx = seg['index']
        adjusted_path = os.path.join(work_dir, f"adj_{idx:04d}.wav")

        if not os.path.exists(adjusted_path):
            continue

        try:
            audio, sr = sf.read(adjusted_path, dtype='float32')
            if len(audio.shape) > 1:
                audio = audio[:, 0]  # mono

            start_sample = int(seg['start'] * SAMPLE_RATE)
            end_sample = min(start_sample + len(audio), total_samples)
            samples_to_write = end_sample - start_sample

            if samples_to_write > 0:
                timeline[start_sample:end_sample] = audio[:samples_to_write]
        except Exception:
            pass

        if (i+1) % 200 == 0 or i == total-1:
            print(f"  Placed: {i+1}/{total}")

    # Normalize
    peak = np.max(np.abs(timeline))
    if peak > 0:
        timeline = timeline / peak * 0.95

    sf.write(output_audio, timeline, SAMPLE_RATE)
    audio_dur = len(timeline) / SAMPLE_RATE
    print(f"  Timeline: {audio_dur:.1f}s audio written to {output_audio}")
    return output_audio


# ============================================================
# Voice Maps
# ============================================================

EDGE_VOICE_MAP = {
    # Languages
    'chinese': 'zh-CN-YunxiNeural', 'zh': 'zh-CN-YunxiNeural',
    'spanish': 'es-ES-AlvaroNeural', 'es': 'es-ES-AlvaroNeural',
    'french': 'fr-FR-HenriNeural', 'fr': 'fr-FR-HenriNeural',
    'japanese': 'ja-JP-KeitaNeural', 'ja': 'ja-JP-KeitaNeural',
    'german': 'de-DE-ConradNeural', 'de': 'de-DE-ConradNeural',
    'italian': 'it-IT-DiegoNeural', 'it': 'it-IT-DiegoNeural',
    'portuguese': 'pt-BR-AntonioNeural', 'pt': 'pt-BR-AntonioNeural',
    'korean': 'ko-KR-InJoonNeural', 'ko': 'ko-KR-InJoonNeural',
    'russian': 'ru-RU-DmitryNeural', 'ru': 'ru-RU-DmitryNeural',
    'arabic': 'ar-SA-HamedNeural', 'ar': 'ar-SA-HamedNeural',
    'hindi': 'hi-IN-MadhurNeural', 'hi': 'hi-IN-MadhurNeural',
    'turkish': 'tr-TR-AhmetNeural', 'tr': 'tr-TR-AhmetNeural',
    'english': 'en-US-BrianNeural', 'en': 'en-US-BrianNeural',
}


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 5:
        print("Usage: sync_tts.py <srt_file> <work_dir> <tts_engine> <target_lang> [voice_profile] [voice_name]")
        print("  tts_engine: edge-tts, kokoro, or voicebox")
        print("  voice_profile: voicebox profile name (required for voicebox)")
        print("  voice_name: specific voice ID override (e.g. en-US-BrianNeural, am_michael)")
        sys.exit(1)

    srt_file = sys.argv[1]
    work_dir = sys.argv[2]
    tts_engine = sys.argv[3]
    target_lang = sys.argv[4]
    voice_profile = sys.argv[5] if len(sys.argv) > 5 else None
    voice_name = sys.argv[6] if len(sys.argv) > 6 else None

    if tts_engine == 'voicebox' and not voice_profile:
        print("Error: voicebox engine requires voice_profile parameter")
        sys.exit(1)

    os.makedirs(work_dir, exist_ok=True)
    output_audio = os.path.join(work_dir, "combined.wav")

    t_global = time.time()

    # Parse SRT
    print("Parsing subtitles...")
    segments = parse_srt(srt_file)
    total = len(segments)
    print(f"Found {total} segments\n")

    # Step 1: Generate TTS
    print(f"=== Step 1: TTS Generation ({tts_engine}) ===")
    t1 = time.time()

    if tts_engine == 'edge-tts':
        voice = voice_name or EDGE_VOICE_MAP.get(target_lang, 'en-US-BrianNeural')
        print(f"Voice: {voice}")
        generate_edge_tts(segments, work_dir, voice)
    elif tts_engine == 'kokoro':
        voice = voice_name or 'am_michael'
        print(f"Voice: {voice}")
        generate_kokoro_tts(segments, work_dir, voice)
    elif tts_engine == 'voicebox':
        print(f"Voice profile: {voice_profile}")
        generate_voicebox_tts(segments, work_dir, voice_profile)

    gen_time = time.time() - t1
    print(f"TTS generation: {gen_time:.1f}s ({gen_time/60:.1f} min)\n")

    # Verify generated files
    missing = []
    for seg in segments:
        idx = seg['index']
        mp3 = os.path.join(work_dir, f"raw_{idx:04d}.mp3")
        wav = os.path.join(work_dir, f"raw_{idx:04d}.wav")
        has_mp3 = os.path.exists(mp3) and os.path.getsize(mp3) > 100
        has_wav = os.path.exists(wav) and os.path.getsize(wav) > 100
        if not has_mp3 and not has_wav:
            missing.append(idx + 1)
    if missing:
        print(f"WARNING: {len(missing)} missing segments: {missing[:10]}...")

    # Step 2: Speed adjustment
    print(f"\n=== Step 2: Speed Adjustment ===")
    t2 = time.time()
    speed_adjust_all(segments, work_dir)
    adj_time = time.time() - t2
    print(f"Speed adjustment: {adj_time:.1f}s\n")

    # Step 3: Build numpy timeline
    print(f"=== Step 3: Building Audio Timeline ===")
    t3 = time.time()
    build_numpy_timeline(segments, work_dir, output_audio)
    build_time = time.time() - t3
    print(f"Timeline built: {build_time:.1f}s\n")

    total_time = time.time() - t_global
    print(f"=== Sync Complete ===")
    print(f"  TTS generation:    {gen_time:.1f}s ({gen_time/60:.1f} min)")
    print(f"  Speed adjustment:  {adj_time:.1f}s")
    print(f"  Timeline building: {build_time:.1f}s")
    print(f"  TOTAL:             {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Output: {output_audio}")

    # Save segments info
    with open(os.path.join(work_dir, 'segments.json'), 'w') as f:
        json.dump(segments, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
