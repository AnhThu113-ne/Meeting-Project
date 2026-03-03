"""
Background AI processor: runs transcription → diarization → Gemini summary → saves to DB.
"""
import os
import sys
import json
import subprocess
import asyncio
import google.generativeai as genai
import requests
import base64
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Path to venv python
VENV_PYTHON = os.path.join(
    os.path.dirname(__file__), "..", "..", "venv", "Scripts", "python.exe"
)
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")



def find_ffmpeg() -> str:
    """Find ffmpeg executable path, checking PATH and common install locations."""
    import shutil

    # 1. Check PATH first
    path = shutil.which("ffmpeg")
    if path:
        return path

    # 2. Check common WinGet install locations
    import glob
    winget_patterns = [
        r"C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin\ffmpeg.exe",
        r"C:\ProgramData\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]
    for pattern in winget_patterns:
        matches = glob.glob(pattern, recursive=False)
        if matches:
            return matches[0]

    raise RuntimeError(
        "FFmpeg not found. Please install FFmpeg: winget install ffmpeg\n"
        "Then restart the server."
    )


# Cache ffmpeg path
_FFMPEG_PATH = None


def get_ffmpeg() -> str:
    global _FFMPEG_PATH
    if _FFMPEG_PATH is None:
        _FFMPEG_PATH = find_ffmpeg()
    return _FFMPEG_PATH


def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to WAV 16kHz mono using FFmpeg.
    Returns path to the converted WAV file.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path  # Already WAV, no conversion needed

    wav_path = os.path.splitext(input_path)[0] + "_converted.wav"
    result = subprocess.run(
        [
            get_ffmpeg(), "-y",    # ← dùng đường dẫn tuyệt đối tự tìm
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-vn",
            wav_path
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        # Check if file simply has no audio track
        if "no streams" in result.stderr or "does not contain" in result.stderr or "Output file is empty" in result.stderr:
            raise RuntimeError("FILE_NO_AUDIO")
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr[-300:]}")

    # Verify output file was created and has content
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1024:
        raise RuntimeError("FILE_NO_AUDIO")

    return wav_path


def check_audio_not_silent(wav_path: str, min_rms: float = 10.0):
    """
    Check that a WAV file is not completely silent.
    Raises RuntimeError with user-friendly message if silent.
    """
    import wave
    import struct
    import math

    try:
        with wave.open(wav_path, 'rb') as w:
            # Read first 3 seconds (3 * 16000 samples)
            n_check = min(w.getnframes(), 16000 * 3)
            raw = w.readframes(n_check)

        if not raw:
            raise RuntimeError("FILE_NO_AUDIO")

        samples = struct.unpack('<' + 'h' * (len(raw) // 2), raw)
        if not samples:
            raise RuntimeError("FILE_NO_AUDIO")

        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        if rms < min_rms:
            raise RuntimeError("FILE_NO_AUDIO")
    except RuntimeError:
        raise
    except Exception:
        pass  # If we can't check, let transcription handle it



async def process_meeting(meeting_id: int, audio_path: str, db_session_factory):
    """Main async processor: runs AI pipeline and saves results to DB."""
    from backend.db.database import Meeting, Transcript, MeetingMinutes

    db = db_session_factory()
    wav_path = audio_path  # will be updated if conversion needed
    try:
        # Mark as processing
        meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if not meeting:
            return
        meeting.status = "processing"
        db.commit()

        # ---- Step 0: Convert to WAV if needed ----
        wav_path = convert_to_wav(audio_path)

        # ---- Step 0b: Verify audio has actual sound ----
        check_audio_not_silent(wav_path)

        # ---- Step 1: Transcription ----
        trans_output = os.path.join(os.path.dirname(audio_path), f"trans_{meeting_id}.json")
        result = subprocess.run(
            [VENV_PYTHON, os.path.join(SCRIPTS_DIR, "run_transcription.py"), wav_path, trans_output],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Transcription failed: {result.stderr[-500:]}")

        # ---- Step 2: Diarization ----
        diar_output = os.path.join(os.path.dirname(audio_path), f"diar_{meeting_id}.json")
        result = subprocess.run(
            [VENV_PYTHON, os.path.join(SCRIPTS_DIR, "run_diarization.py"), wav_path, diar_output],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Diarization failed: {result.stderr[-500:]}")

        # ---- Step 3: Merge + Save transcripts ----
        with open(trans_output, "r", encoding="utf-8") as f:
            trans_data = json.load(f)
        with open(diar_output, "r", encoding="utf-8") as f:
            diar_data = json.load(f)

        full_text = ""
        for t in trans_data:
            speaker = "Unknown"
            for d in diar_data:
                if d["start"] <= t["start"] <= d["end"]:
                    speaker = d["speaker"]
                    break
            seg = Transcript(
                meeting_id=meeting_id,
                speaker=speaker,
                text=t["text"].strip(),
                start_time=t["start"],
                end_time=t["end"]
            )
            db.add(seg)
            full_text += f"{speaker}: {t['text']}\n"
        db.commit()

        # Update duration
        if trans_data:
            meeting.duration = trans_data[-1]["end"]
            db.commit()

        # ---- Step 4: Gemini Summary ----
        summary_text = full_text[:3000]  # fallback
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = (
                "Hay tao bien ban cuoc hop chuyen nghiep tu noi dung sau. "
                "Bao gom: tom tat, ket luan va cong viec can lam (neu co).\n\n"
                + full_text[:6000]
            )
            resp = model.generate_content(prompt)
            summary_text = resp.text
        except Exception as e:
            summary_text = f"[Gemini error: {e}]\n\n{full_text[:2000]}"

        minutes = MeetingMinutes(
            meeting_id=meeting_id,
            summary=summary_text
            # Khong luu raw_transcript - da co bang transcripts rieng
        )
        db.add(minutes)

        meeting.status = "completed"
        db.commit()

        # Cleanup temp files (transcription/diarization JSON)
        cleanup = [trans_output, diar_output]
        if wav_path != audio_path:
            cleanup.append(wav_path)  # Remove converted WAV
        for f in cleanup:
            if os.path.exists(f):
                os.remove(f)

        # Xoa file upload goc de tiet kiem dung luong o dia
        # (Da luu het ket qua vao DB, khong can giu file nua)
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass  # Khong loi neu khong xoa duoc

    except Exception as e:
        db.rollback()
        error_msg = str(e)
        # Friendly messages for known errors
        if "FILE_NO_AUDIO" in error_msg:
            error_msg = "error: File nay khong co am thanh. Vui long up len file co ghi am (WAV, MP3, M4A) hoac video co tieng."
        elif "Transcription failed" in error_msg:
            error_msg = "error: Khong the doc am thanh. Thu lai voi file khac."
        elif "FFmpeg" in error_msg:
            error_msg = "error: Dinh dang file khong duoc ho tro. Dung WAV, MP3, M4A, AAC hoac MP4."
        else:
            error_msg = f"error: {error_msg[:150]}"
        meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if meeting:
            meeting.status = error_msg
            db.commit()
    finally:
        # Cleanup converted WAV even on error
        if wav_path != audio_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
        db.close()

def generate_tts_viettel(text: str) -> bytes:
    """Gọi Viettel TTS để tạo file âm thanh lời chào"""
    API_KEY = os.getenv("VIETTEL_API_KEY", "") # Nhớ thêm vào .env
    url = "https://viettelgroup.ai/voice/api/tts/v1/rest/syn"
    headers = {"token": API_KEY, "Content-Type": "application/json"}
    data = {
        "text": text,
        "voice": "hn-female",
        "speed": 1.0,
        "tts_return_option": "2" # Trả về base64
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            audio_base64 = response.json().get("audio_content")
            return base64.b64decode(audio_base64)
    except Exception as e:
        print(f"TTS Error: {e}")
    return b""
