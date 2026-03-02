"""
Speaker Enrollment: tao voice profile tu file audio mau 30s
Luu vao dataset/speaker_profiles.json
"""
import sys
import os
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Them thu muc goc vao sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.run_diarization import compute_mfcc

PROFILES_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "speaker_profiles.json")


def extract_voice_profile(audio_path: str) -> list:
    """Trich xuat MFCC mean vector dai dien cho giong noi."""
    from scipy.io import wavfile
    from sklearn.preprocessing import StandardScaler

    sr, wav_int = wavfile.read(audio_path)
    wav = wav_int.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav / (np.max(np.abs(wav)) + 1e-8)

    if sr != 16000:
        from scipy.signal import resample
        wav = resample(wav, int(len(wav) * 16000 / sr))
        sr = 16000

    mfcc = compute_mfcc(wav, sr, n_mfcc=20)
    # Profile = mean + std cua toan bo file
    profile = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])
    return profile.tolist()


def load_profiles() -> dict:
    if os.path.exists(PROFILES_PATH):
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_profiles(profiles: dict):
    os.makedirs(os.path.dirname(PROFILES_PATH), exist_ok=True)
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


def enroll_speaker(name: str, audio_path: str):
    """Dang ky mot nguoi dung vao database giong noi."""
    print(f"Enrolling '{name}' from: {audio_path}")
    profile = extract_voice_profile(audio_path)
    profiles = load_profiles()
    profiles[name] = {"profile": profile, "audio_path": audio_path}
    save_profiles(profiles)
    print(f"Done! Saved profile for '{name}'. Total speakers: {len(profiles)}")
    return profile


def enroll_all_members():
    """Dang ky tat ca thanh vien tu thu muc dataset/members/."""
    members_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "members")
    name_map = {
        "thu.wav": "Thu",
        "my.wav": "My",
        "Van.wav": "Van",
    }
    for filename, name in name_map.items():
        path = os.path.join(members_dir, filename)
        if os.path.exists(path):
            enroll_speaker(name, path)
        else:
            print(f"Warning: File not found: {path}")
    return load_profiles()


def identify_speaker(segment_profile: np.ndarray, profiles: dict, threshold: float = 0.85) -> str:
    """
    So sanh segment_profile voi tat ca profiles da dang ky.
    Tra ve ten neu do tuong dong >= threshold, nguoc lai tra ve 'Unknown'.
    """
    if not profiles:
        return None

    best_name = None
    best_sim = -1

    seg = np.array(segment_profile)
    seg_norm = seg / (np.linalg.norm(seg) + 1e-8)

    for name, data in profiles.items():
        ref = np.array(data["profile"])
        ref_norm = ref / (np.linalg.norm(ref) + 1e-8)
        similarity = float(np.dot(seg_norm, ref_norm))
        if similarity > best_sim:
            best_sim = similarity
            best_name = name

    if best_sim >= threshold:
        return best_name
    return "Unknown"


if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Usage: python enroll.py "Ten Nguoi" "path/to/audio.wav"
        enroll_speaker(sys.argv[1], sys.argv[2])
    else:
        # Enroll tat ca thanh vien mac dinh
        print("Enrolling all members from dataset/members/...")
        enroll_all_members()
        profiles = load_profiles()
        print(f"\nDone! Registered {len(profiles)} speakers:")
        for name in profiles:
            print(f"  - {name}")
