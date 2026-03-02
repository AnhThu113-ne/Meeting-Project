"""
Viettel AI Speech-to-Text API
Endpoint: POST https://viettelai.vn/asr/recognize
Auth: token trong multipart form-data

Fallback: neu Viettel API loi → dung Faster-Whisper
"""
import sys
import os
import json
import requests

def transcribe_with_viettel(audio_path: str, output_path: str, api_key: str) -> bool:
    """
    Goi Viettel STT API de chuyen audio → text.
    Returns True neu thanh cong, False neu can fallback.
    """
    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                "https://viettelai.vn/asr/recognize",
                headers={"accept": "*/*"},
                data={"token": api_key},
                files={"file": ("audio.wav", f, "audio/wav")},
                timeout=120
            )

        if resp.status_code == 200:
            data = resp.json()
            # Viettel tra ve: {"code": 200, "message": "...", "response": {...}}
            if data.get("code") == 200:
                response_data = data.get("response", {})
                # Extract transcript text
                text = ""
                if isinstance(response_data, dict):
                    text = response_data.get("transcript", response_data.get("text", str(response_data)))
                elif isinstance(response_data, str):
                    text = response_data

                # Format ket qua giong nhu Whisper (list of segments)
                results = [{"start": 0.0, "end": 0.0, "text": text}]
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False)
                print(f"[Viettel STT] Thanh cong → {output_path}")
                return True
            else:
                print(f"[Viettel STT] API error: {data.get('message', 'Unknown')}")
                return False
        else:
            print(f"[Viettel STT] HTTP {resp.status_code}: {resp.text[:200]}")
            return False

    except Exception as e:
        print(f"[Viettel STT] Exception: {e}")
        return False


def transcribe_with_whisper(audio_path: str, output_path: str) -> None:
    """Fallback: dung Faster-Whisper neu Viettel API that bai."""
    from faster_whisper import WhisperModel
    print(f"[Whisper Fallback] Processing: {audio_path}")
    # tiny: nhanh hon base ~3x, du tot voi tieng Viet co ban
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, beam_size=3, language="vi")
    results = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[Whisper Fallback] Xong → {output_path}")


def run_transcription(audio_path: str, output_path: str = "transcription_results.json"):
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("VIETTEL_API_KEY", "")

    if api_key:
        print(f"[Transcription] Dang dung Viettel AI STT...")
        success = transcribe_with_viettel(audio_path, output_path, api_key)
        if success:
            return

    # Fallback sang Whisper neu Viettel that bai
    print(f"[Transcription] Fallback sang Faster-Whisper...")
    transcribe_with_whisper(audio_path, output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_transcription.py <audio_path> [output_path]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "transcription_results.json"
    run_transcription(sys.argv[1], out)
