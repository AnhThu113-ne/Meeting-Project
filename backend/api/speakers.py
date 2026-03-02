"""
API Router: /api/speakers - Quan ly dang ky giong noi
"""
import os
import asyncio
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from datetime import datetime

router = APIRouter(prefix="/api/speakers", tags=["speakers"])

MEMBERS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "members")
os.makedirs(MEMBERS_DIR, exist_ok=True)

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
VENV_PYTHON = os.path.join(os.path.dirname(__file__), "..", "..", "venv", "Scripts", "python.exe")


def get_enrollment_module():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from scripts.speaker_enrollment import load_profiles, enroll_speaker
    return load_profiles, enroll_speaker


@router.get("")
def list_speakers():
    """Lay danh sach tat ca nguoi da dang ky giong noi."""
    load_profiles, _ = get_enrollment_module()
    profiles = load_profiles()
    return [
        {"name": name, "audio_path": data.get("audio_path", "")}
        for name, data in profiles.items()
    ]


@router.post("/enroll")
async def enroll_speaker_endpoint(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    """Dang ky giong noi moi: upload file 30s va dat ten."""
    if not name.strip():
        raise HTTPException(400, "Ten khong duoc de trong")

    ext = os.path.splitext(file.filename)[1].lower()
    allowed = {".wav", ".mp3", ".m4a", ".aac", ".ogg"}
    if ext not in allowed:
        raise HTTPException(400, f"Dinh dang {ext} khong ho tro")

    # Luu file
    safe_name = name.strip().replace(" ", "_")
    filename = f"{safe_name}{ext}"
    file_path = os.path.join(MEMBERS_DIR, filename)

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Convert to WAV neu can
    wav_path = file_path
    if ext != ".wav":
        import subprocess, glob
        ffmpeg_candidates = glob.glob(
            r"C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin\ffmpeg.exe"
        )
        ffmpeg = ffmpeg_candidates[0] if ffmpeg_candidates else "ffmpeg"
        wav_path = os.path.splitext(file_path)[0] + ".wav"
        subprocess.run(
            [ffmpeg, "-y", "-i", file_path, "-ar", "16000", "-ac", "1", "-vn", wav_path],
            capture_output=True
        )

    # Trich xuat va luu profile
    try:
        load_profiles, enroll_fn = get_enrollment_module()
        enroll_fn(name.strip(), wav_path)
    except Exception as e:
        raise HTTPException(500, f"Loi dang ky: {e}")

    return {"message": f"Da dang ky thanh cong cho '{name}'", "name": name}


@router.post("/enroll-defaults")
def enroll_default_members():
    """Dang ky tat ca thanh vien mac dinh (Thu, My, Van) tu dataset/members/."""
    import subprocess
    result = subprocess.run(
        [VENV_PYTHON, os.path.join(SCRIPTS_DIR, "speaker_enrollment.py")],
        capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), "..", "..")
    )
    if result.returncode != 0:
        raise HTTPException(500, result.stderr[-200:])
    load_profiles, _ = get_enrollment_module()
    profiles = load_profiles()
    return {"message": f"Da dang ky {len(profiles)} nguoi", "speakers": list(profiles.keys())}


@router.delete("/{name}")
def delete_speaker(name: str):
    """Xoa mot nguoi khoi database giong noi."""
    from scripts.speaker_enrollment import load_profiles, save_profiles
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from scripts.speaker_enrollment import load_profiles, save_profiles
    profiles = load_profiles()
    if name not in profiles:
        raise HTTPException(404, f"Khong tim thay '{name}'")
    del profiles[name]
    save_profiles(profiles)
    return {"message": f"Da xoa '{name}'"}
