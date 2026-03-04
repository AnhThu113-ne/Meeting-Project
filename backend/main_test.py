"""
main_test.py - Backend test mode (KHÔNG cần pyannote/torch/pydub)
Mô phỏng Viettel STT bằng cách nhận text thủ công qua API
Dùng để test luồng: trigger word → ghi log → tóm tắt
"""
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os, shutil, uuid, json, asyncio
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
RESULT_DIR      = "results"
os.makedirs("uploads", exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Import DB và Trigger Detector (không cần AI models)
import database as db
from trigger_detector import check_start_trigger, check_stop_trigger

from llm_processor import MeetingLLMProcessor
llm_proc = MeetingLLMProcessor(api_key=GEMINI_API_KEY)

from audio_processor import MeetingAudioProcessor
audio_processor = MeetingAudioProcessor(
    hf_token=os.getenv("HF_TOKEN"), 
    groq_key=os.getenv("GROQ_API_KEY")
)

app = FastAPI(title="Meeting Minutes API [TEST MODE]")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.mount("/static/voice", StaticFiles(directory=db.VOICE_DIR), name="voice")

# State machine cho moi phong hop
room_states: dict = {}

# ================================================================
@app.get("/")
def root():
    return {"message": "Meeting Minutes API - TEST MODE", "status": "running"}

@app.get("/test", response_class=HTMLResponse)
def test_console():
    """Serve trang Test Console UI"""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_console.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()

# ================================================================
# TEST: Gia lap STT bang cach nhan text thuc tiep (thay Viettel STT)
# ================================================================
@app.post("/test-speak")
async def test_speak(
    background_tasks: BackgroundTasks,
    room_id: str = Form(...),
    text: str    = Form(...)
):
    """
    Gia lap nguoi noi: Nhan van ban thay vi am thanh.
    Dung de test trigger words va luong luu DB ma khong can Viettel STT that.
    """
    if room_id not in room_states:
        room_states[room_id] = {"state": "standby", "db_meeting_id": None, "lines": []}

    state = room_states[room_id]["state"]

    if state == "standby":
        print(f"[{room_id}][STANDBY] '{text}'")
        if check_start_trigger(text):
            mid = db.create_meeting(meeting_code=f"{room_id}_{uuid.uuid4().hex[:4]}")
            room_states[room_id]["state"]         = "recording"
            room_states[room_id]["db_meeting_id"] = mid
            room_states[room_id]["lines"]         = []
            print(f"[{room_id}] >>> BAT DAU GHI! Meeting DB ID={mid}")
            return {"state": "recording", "event": "MEETING_STARTED", "db_id": mid}

    elif state == "recording":
        mid = room_states[room_id]["db_meeting_id"]
        print(f"[{room_id}][RECORDING] '{text}'")

        # Luu vao DB va logs/text
        db.save_transcript_line(meeting_id=mid, text=text, speaker_name="Speaker")
        room_states[room_id]["lines"].append({"speaker": "Speaker", "text": text})

        if check_stop_trigger(text):
            room_states[room_id]["state"] = "finalizing"
            background_tasks.add_task(finalize, room_id, mid)
            return {"state": "finalizing", "event": "MEETING_ENDED"}

        return {"state": "recording", "event": "LINE_SAVED", "text": text}

    elif state == "finalizing":
        return {"state": "finalizing", "event": "PROCESSING"}

    return {"state": state, "room_id": room_id}


def finalize(room_id: str, meeting_id: int):
    """Goi Gemini tao bien ban va luu tat ca. (sync - chay trong BackgroundTasks)"""
    import traceback
    try:
        transcript = room_states.get(room_id, {}).get("lines", [])
        print(f"[{room_id}] Bat dau finalize: {len(transcript)} dong transcript")

        if not transcript:
            print(f"[{room_id}] WARN: transcript rong, tao bien ban tu noi dung mac dinh")
            transcript = [{"speaker": "System", "text": "Cuoc hop da ket thuc (khong co noi dung)"}]

        minutes_md = llm_proc.generate_minutes(transcript)
        print(f"[{room_id}] Gemini tra ve {len(minutes_md)} ky tu")

        path = db.save_meeting_minutes(meeting_id, minutes_md)
        db.end_meeting(meeting_id)

        result_path = os.path.join(RESULT_DIR, f"{room_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({
                "transcript": transcript,
                "minutes": minutes_md,
                "file": path
            }, f, ensure_ascii=False, indent=2)

        print(f"[{room_id}] Da luu bien ban: {result_path}")

    except Exception as e:
        print(f"[{room_id}] LOI FINALIZE: {e}")
        traceback.print_exc()
        # Luu file bao loi de frontend hien thi
        try:
            err_path = os.path.join(RESULT_DIR, f"{room_id}.json")
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump({
                    "transcript": room_states.get(room_id, {}).get("lines", []),
                    "minutes": f"# Lỗi tạo biên bản\n\nLỗi: {str(e)}\n\nVui lòng kiểm tra API key Gemini.",
                    "error": str(e)
                }, f, ensure_ascii=False, indent=2)
        except:
            pass
    finally:
        room_states[room_id] = {"state": "standby", "db_meeting_id": None, "lines": []}
        print(f"[{room_id}] Reset ve standby")


# ================================================================
# ENDPOINTS CHO CHROME EXTENSION (WEB SPEECH API)
# ================================================================
class StartMeetingRequest(BaseModel):
    room_id: str

class AddLineRequest(BaseModel):
    meeting_id: int
    text: str
    speaker: str = "Speaker"

class FinalizeMeetingRequest(BaseModel):
    room_id: str
    meeting_id: int
    transcript: List[Dict[str, Any]]

@app.post("/start-meeting")
def api_start_meeting(req: StartMeetingRequest):
    mid = db.create_meeting(meeting_code=f"{req.room_id}_{uuid.uuid4().hex[:4]}")
    room_states[req.room_id] = {
        "state": "recording",
        "db_meeting_id": mid,
        "lines": []
    }
    return {"status": "ok", "meeting_id": mid}

@app.post("/add-line")
def api_add_line(req: AddLineRequest):
    db.save_transcript_line(meeting_id=req.meeting_id, text=req.text, speaker_name=req.speaker)
    return {"status": "ok"}

@app.post("/finalize-meeting")
def api_finalize_meeting(req: FinalizeMeetingRequest, background_tasks: BackgroundTasks):
    room_states[req.room_id] = {
        "state": "finalizing",
        "db_meeting_id": req.meeting_id,
        "lines": req.transcript
    }
    background_tasks.add_task(finalize, req.room_id, req.meeting_id)
    return {"status": "finalizing"}

@app.post("/upload-voice")
async def api_upload_voice(meeting_id: int = Form(...), file: UploadFile = File(...)):
    """Lưu trữ file âm thanh của cuộc họp."""
    ext = os.path.splitext(file.filename)[1] or ".webm"
    # Ghi đè hoặc append vào chung 1 file duy nhất cho 1 meeting thay vì nhiều file UUID
    path = os.path.join(db.VOICE_DIR, f"meeting_{meeting_id}_audio_record{ext}")
    
    with open(path, "ab") as f:
        f.write(await file.read())
        
    db.update_meeting_paths(meeting_id, audio_path=path)
    print(f"[DB] Da luu audio chunk vao: {path}")
    return {"status": "ok", "path": path}

@app.post("/upload-audio")
async def api_upload_audio(file: UploadFile = File(...)):
    """API dùng cho Frontend Dashboard để phân tích file offline (Diarization + Voice Biometrics)."""
    temp_path = os.path.join("uploads", f"offline_{uuid.uuid4().hex[:6]}_{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    print(f"\n[AI] Bắt đầu xử lý Biometrics cho file: {file.filename}")
    # Lấy các mẫu giọng đăng ký từ SQL Server
    speaker_refs = db.get_speaker_voice_paths()
    
    # Process Phân mảnh + Nhận diện giọng nói + Bóc băng
    transcript = audio_processor.process_audio(temp_path, references=speaker_refs)
    
    # Tạo biên bản từ text
    minutes_content = "Không tạo được do chưa nhận diện được text."
    if transcript:
        try:
            minutes_content = llm_proc.generate_minutes(transcript)
        except Exception as e:
            minutes_content = f"Lỗi tạo biên bản: {str(e)}"

    return {
        "status": "ok",
        "transcript": transcript,
        "minutes": minutes_content
    }

# ================================================================
# TEST & UTILS
@app.get("/room-state/{room_id}")
def get_room_state(room_id: str):
    """Xem trang thai phong hop hien tai."""
    return room_states.get(room_id, {"state": "not_found"})

@app.get("/result/{room_id}")
def get_result(room_id: str):
    """Lay bien ban sau khi xu ly xong."""
    p = os.path.join(RESULT_DIR, f"{room_id}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"status": "not_ready"}

@app.get("/meetings")
def get_meetings():
    """Lấy danh sách tất cả các cuộc họp từ hệ thống."""
    return db.get_all_meetings()

@app.get("/meeting/{meeting_id}/details")
def get_meeting_details(meeting_id: int):
    """Lấy chi tiết transcript và nội dung biên bản cuộc họp."""
    transcript = db.get_full_transcript(meeting_id)
    
    # Lấy đường dẫn file biên bản
    all_meetings = db.get_all_meetings()
    current = next((m for m in all_meetings if m["id"] == meeting_id), None)
    
    minutes_content = "Không tìm thấy biên bản hoặc đang trong quá trình tạo."
    if current and current["minutes_path"] and os.path.exists(current["minutes_path"]):
        with open(current["minutes_path"], "r", encoding="utf-8") as f:
            minutes_content = f.read()
            
    return {
        "meeting_id": meeting_id,
        "transcript": transcript,
        "minutes": minutes_content,
        "audio_path": current.get("audio_path") if current else None,
        "transcript_path": current.get("transcript_path") if current else None
    }

@app.get("/speakers")
def list_speakers():
    return db.get_all_speakers()

@app.post("/register-speaker")
async def register_speaker(name: str = Form(...), file: UploadFile = File(...)):
    ext      = os.path.splitext(file.filename)[1] or ".wav"
    filename = f"{name.strip().replace(' ','_')}_{uuid.uuid4().hex[:6]}{ext}"
    path     = os.path.join(db.VOICE_DIR, filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    sid = db.register_speaker(name=name, voice_file_path=path)
    return {"status": "ok", "speaker_id": sid, "name": name, "path": path}

if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print(" MEETING MINUTES - TEST MODE")
    print(" Swagger UI: http://localhost:8000/docs")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
