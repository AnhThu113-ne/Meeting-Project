"""
Speaker Diarization using scipy WAV loading + MFCC via numpy + KMeans.
Avoids librosa/numba issues. Pure Python approach.
Co tich hop Speaker Enrollment: gan ten that thay vi SPEAKER_XX.
"""
import sys
import os
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def compute_mfcc(wav, sr, n_mfcc=20, frame_length=400, hop_length=160):
    """Compute MFCC features using numpy (no librosa/numba needed)."""
    # Pre-emphasis
    wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])
    
    # Frame the signal
    num_frames = 1 + (len(wav) - frame_length) // hop_length
    frames = np.stack([wav[i * hop_length: i * hop_length + frame_length]
                       for i in range(num_frames)])
    
    # Window
    window = np.hamming(frame_length)
    frames = frames * window
    
    # FFT power spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)
    
    # Mel filter bank
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mfcc + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((NFFT + 1) * hz_points / sr).astype(int)
    
    fbank = np.zeros((n_mfcc, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, n_mfcc + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    
    # DCT (MFCC)
    from scipy.fft import dct
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    return mfcc


def diarize_audio(audio_path, output_json="diarization_results.json"):
    print(f"--- [Diarization] Processing: {audio_path} ---")

    from scipy.io import wavfile
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    # Load WAV
    sr, wav_int = wavfile.read(audio_path)
    # Convert to float
    wav = wav_int.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav / np.max(np.abs(wav) + 1e-8)

    # Resample to 16kHz if needed
    if sr != 16000:
        from scipy.signal import resample
        target_len = int(len(wav) * 16000 / sr)
        wav = resample(wav, target_len)
        sr = 16000

    total_sec = len(wav) / sr
    print(f"Audio duration: {total_sec:.1f}s, SR: {sr}Hz")

    # Sliding window
    seg_duration = 1.5
    hop_duration = 0.75
    seg_len = int(seg_duration * sr)
    hop_len = int(hop_duration * sr)

    segments = []
    features = []
    t = 0
    while t + seg_len <= len(wav):
        chunk = wav[t: t + seg_len]
        mfcc = compute_mfcc(chunk, sr, n_mfcc=20)
        feat = np.concatenate([
            mfcc.mean(axis=0),
            mfcc.std(axis=0)
        ])
        features.append(feat)
        segments.append({
            "start": round(t / sr, 2),
            "end": round((t + seg_len) / sr, 2)
        })
        t += hop_len

    if len(segments) < 2:
        result = [{"start": 0.0, "end": round(total_sec, 2), "speaker": "SPEAKER_00"}]
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Diarization saved -> {output_json}")
        return result

    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Auto-detect number of speakers
    best_k, best_score = 1, -1
    max_k = min(6, len(segments) - 1)
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            print(f"  k={k}: silhouette={score:.3f}")
            if score > best_score:
                best_score = score
                best_k = k

    print(f"Detected {best_k} speakers (best silhouette={best_score:.3f})")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    for i, seg in enumerate(segments):
        seg["speaker"] = f"SPEAKER_{labels[i]:02d}"

    # Merge consecutive same-speaker segments
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        if seg["speaker"] == merged[-1]["speaker"] and seg["start"] <= merged[-1]["end"] + 0.5:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))

    # ---- Gan ten that neu co Enrollment ----
    try:
        # Import enrollment module
        enroll_dir = os.path.dirname(__file__)
        sys.path.insert(0, os.path.join(enroll_dir, ".."))
        from scripts.speaker_enrollment import (
            load_profiles, extract_voice_profile, identify_speaker
        )
        profiles = load_profiles()
        if profiles:
            print(f"\nFound {len(profiles)} enrolled speaker(s): {list(profiles.keys())}")
            # Gan ten cho tung cluster
            cluster_to_name = {}
            for cluster_id in range(best_k):
                # Lay cac segment thuoc cluster nay
                cluster_idxs = [i for i, s in enumerate(segments) if s["speaker"] == f"SPEAKER_{cluster_id:02d}"]
                if not cluster_idxs:
                    continue
                # Lay MFCC feature trung binh cua cluster
                cluster_feat = np.array([features[i] for i in cluster_idxs]).mean(axis=0)
                name = identify_speaker(cluster_feat, profiles, threshold=0.6)
                cluster_to_name[f"SPEAKER_{cluster_id:02d}"] = name if name else f"SPEAKER_{cluster_id:02d}"

            # Cap nhat ten trong segments
            for seg in merged:
                old_label = seg["speaker"]
                seg["speaker"] = cluster_to_name.get(old_label, old_label)
            print("Name mapping:", cluster_to_name)
        else:
            print("No enrolled speakers found. Using SPEAKER_XX labels.")
    except Exception as e:
        print(f"Enrollment matching skipped: {e}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\nDiarization complete -> {output_json}")
    for seg in merged:
        print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}")

    return merged


if __name__ == "__main__":
    if len(sys.argv) < 2:
        audio_file = "dataset/meetings/test_audio.wav"
    else:
        audio_file = sys.argv[1]

    out_file = sys.argv[2] if len(sys.argv) > 2 else "diarization_results.json"

    if not os.path.exists(audio_file):
        print(f"ERROR: File not found: {audio_file}")
        sys.exit(1)

    diarize_audio(audio_file, out_file)
