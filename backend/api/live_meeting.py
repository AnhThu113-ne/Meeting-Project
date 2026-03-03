import os
import tempfile
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
from datetime import datetime

from scripts.run_transcription import transcribe_with_whisper # Dùng Whisper tiny cho nhanh (hoặc viết hàm nhận Viettel STT trực tiếp từ bytes)
from scripts.speaker_enrollment import load_profiles, identify_speaker, extract_voice_profile
from backend.core.processor import generate_tts_viettel
from backend.db.database import Meeting, SessionLocal
from backend.api.meetings import run_ai_pipeline

router = APIRouter(prefix="/api/live", tags=["live"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "uploads")

@router.websocket("/stream")
async def live_meeting_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    state = "WAITING_GREETING"
    meeting_audio_path = ""
    audio_buffer = bytearray()
    
    profiles = load_profiles()
    
    try:
        while True:
            # Nhận chunk audio từ Frontend (trình duyệt gửi lên)
            # Khuyến nghị Frontend gửi từng chunk 2-3 giây (định dạng WAV)
            data = await websocket.receive_bytes()
            
            if state in ["WAITING_GREETING", "WAITING_START"]:
                # Lưu chunk tạm để nhận diện
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                
                # 1. Gọi AI chuyển Speech to Text (chỉ lấy text của 3s này)
                # Lưu ý: Cần viết 1 hàm quick_stt đọc file trả về text string
                text = quick_transcribe_for_live(tmp_path).lower() 
                
                if state == "WAITING_GREETING" and "chào" in text:
                    # 2. Ai đang nói? Trích xuất MFCC
                    try:
                        mfcc_profile = extract_voice_profile(tmp_path)
                        name = identify_speaker(mfcc_profile, profiles, threshold=0.6)
                        if name == "Unknown":
                            name = "bạn"
                    except:
                        name = "bạn"
                        
                    # 3. AI đáp lời
                    reply_text = f"Chào {name}. Bạn đã sẵn sàng họp chưa?"
                    audio_reply = generate_tts_viettel(reply_text)
                    
                    # Gửi âm thanh về cho Frontend phát lên loa
                    await websocket.send_bytes(audio_reply)
                    state = "WAITING_START"
                    await websocket.send_json({"status": "greetings_done", "message": reply_text})
                    
                elif state == "WAITING_START" and "bắt đầu" in text:
                    # Chuyển sang ghi âm toàn bộ
                    state = "RECORDING"
                    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    meeting_audio_path = os.path.join(UPLOAD_DIR, f"live_{timestamp}.webm")
                    
                    audio_reply = generate_tts_viettel("Đã hiểu. Tôi bắt đầu ghi lại cuộc họp từ bây giờ.")
                    await websocket.send_bytes(audio_reply)
                    await websocket.send_json({"status": "recording_started"})
                
                os.unlink(tmp_path) # Xóa file nháp 3s
                
            elif state == "RECORDING":
                # Bắt đầu gom âm thanh vào file chính
                with open(meeting_audio_path, "ab") as f:
                    f.write(data)
                
    except WebSocketDisconnect:
        # Khi User tắt trình duyệt hoặc bấm nút "Kết thúc họp"
        print("Client disconnected.")
        if state == "RECORDING" and os.path.exists(meeting_audio_path):
            # Lưu vào DB và kích hoạt luồng xử lý bạn đã viết sẵn!
            db = SessionLocal()
            meeting = Meeting(
                title=f"Live Meeting {datetime.now().strftime('%d/%m/%Y')}",
                filename=os.path.basename(meeting_audio_path),
                file_path=meeting_audio_path,
                status="pending"
            )
            db.add(meeting)
            db.commit()
            db.refresh(meeting)
            db.close()
            
            # Kích hoạt luồng AI chạy ngầm
            import threading
            threading.Thread(target=run_ai_pipeline, args=(meeting.id, meeting_audio_path)).start()

def quick_transcribe_for_live(audio_path):
    """Hàm STT siêu nhanh cho nhận diện khẩu lệnh (dùng Whisper tiny)"""
    from faster_whisper import WhisperModel
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language="vi")
    return " ".join([s.text for s in segments])