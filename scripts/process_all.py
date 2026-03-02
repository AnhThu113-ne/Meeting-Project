import subprocess
import os
import json
import sys
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def run_workflow(audio_path):
    print(f"🚀 Bắt đầu Workflow đa tiến trình cho: {audio_path}")
    
    # 1. Chạy Transcription
    print("\n[Vòng 1] Chuyển giọng nói sang văn bản...")
    subprocess.run([sys.executable, "scripts/run_transcription.py", audio_path], check=True)
    
    # 2. Chạy Diarization
    print("\n[Vòng 2] Phân tách người nói...")
    # Lưu ý: Cần gỡ bỏ openai-whisper trước khi chạy Diarization nếu vẫn lỗi
    subprocess.run([sys.executable, "scripts/run_diarization.py", audio_path], check=True)
    
    # 3. Đọc kết quả và gộp
    if os.path.exists("transcription_results.json") and os.path.exists("diarization_results.json"):
        with open("transcription_results.json", "r") as f:
            trans = json.load(f)
        with open("diarization_results.json", "r") as f:
            diar = json.load(f)
            
        print("\n--- KẾT QUẢ KẾT HỢP ---")
        full_text = ""
        for t in trans:
            # Tìm speaker tương ứng (đơn giản hóa: lấy speaker có thời gian trùng khớp nhất)
            speaker = "Unknown"
            for d in diar:
                if d["start"] <= t["start"] <= d["end"]:
                    speaker = d["speaker"]
                    break
            print(f"[{speaker}]: {t['text']}")
            full_text += f"{speaker}: {t['text']}\n"
            
        # 4. Tóm tắt với Gemini
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash')
            prompt = f"Hay tom tat cuoc hop nay chuyen nghiep tu transcript sau:\n\n{full_text}"
            response = model.generate_content(prompt)
            print("\n[BIEN BAN TOM TAT]:\n", response.text)
        except Exception as e:
            print(f"\n[GEMINI ERROR]: {e}")
            print("\n[RAW TRANSCRIPT]:\n", full_text[:2000])
    else:
        print("❌ Lỗi: Không tìm thấy file kết quả trung gian.")

if __name__ == "__main__":
    audio_file = "dataset/meetings/test_audio.wav"
    if os.path.exists(audio_file):
        run_workflow(audio_file)
    else:
        print(f"❌ Không tìm thấy: {audio_file}")
