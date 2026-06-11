from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import sys
import shutil
import uuid
import json
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

# Reconfigure console stream to UTF-8 to prevent encoding crashes on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Import các Processor và Database
from audio_processor import MeetingAudioProcessor
from llm_processor import MeetingLLMProcessor
import database as db
from trigger_detector import check_start_trigger, check_stop_trigger

# Tải cấu hình từ .env
load_dotenv()

HF_TOKEN       = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VIETTEL_STT_KEY= os.getenv("VIETTEL_STT_KEY")
SQL_SERVER     = os.getenv("SQL_SERVER", "localhost")
SQL_DATABASE   = os.getenv("SQL_DATABASE", "MeetingMinutesDB")

app = FastAPI(title="Meeting Minutes Generator API")

# Khởi tạo các processor
audio_proc = MeetingAudioProcessor(hf_token=HF_TOKEN, viettel_key=VIETTEL_STT_KEY)
llm_proc = MeetingLLMProcessor(api_key=GEMINI_API_KEY)

# Thư mục lưu kết quả
RESULT_DIR = "results"
for d in ["uploads", RESULT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# Quản lý trạng thái các phiên họp real-time
# Cấu trúc: {
#   meeting_code: {
#     "state": "standby" | "recording" | "finalizing",
#     "db_meeting_id": int | None,
#     "lines": [...]      # transcript dòng hội thoại
#   }
# }
room_states: dict = {}
live_sessions: dict = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/voice", StaticFiles(directory=db.VOICE_DIR), name="voice")

async def process_meeting(file_path: str, file_id: str, meeting_id: int = None):
    try:
        # Lấy giọng mẫu từ SQL Server để nhận diện tên người nói
        references = db.get_speaker_voice_paths()

        # Bước 1: Speaker Diarization + Viettel STT
        print(f"Bat dau xu ly am thanh: {file_path}")
        transcript = audio_proc.process_audio(file_path, references=references if references else None)

        # Bước 2: Lưu từng dòng transcript vào SQL Server (file text vào logs/text)
        if meeting_id:
            for turn in transcript:
                db.save_transcript_line(
                    meeting_id  = meeting_id,
                    text        = turn["text"],
                    speaker_name= turn.get("speaker"),
                    start_sec   = turn.get("start"),
                    end_sec     = turn.get("end")
                )

        # Bước 3: Tóm tắt bằng Gemini
        print("Bat dau tom tat bang LLM...")
        minutes_markdown = llm_proc.generate_minutes(transcript)

        # Bước 4: Lưu biên bản vào logs/minutes + SQL Server
        minutes_path = db.save_meeting_minutes(meeting_id, minutes_markdown) if meeting_id else None
        if meeting_id:
            db.end_meeting(meeting_id)

        # Bước 5: Lưu JSON kết quả cho Frontend
        result_data = {"file_id": file_id, "transcript": transcript, "minutes": minutes_markdown}
        with open(os.path.join(RESULT_DIR, f"{file_id}.json"), "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

        print(f"Xu ly hoan tat cho ID: {file_id}")
    except Exception as e:
        print(f"Loi khi xu ly: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Meeting Minutes API is Running"}

# =====================================================================
# ENDPOINT: Đăng ký giọng nói mẫu (lưu file vào logs/voice)
# =====================================================================
@app.post("/register-speaker")
async def register_speaker(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    """Nhận file giọng mẫu, tự động chuyển đổi sang WAV chuẩn mono 16kHz, lưu vào logs/voice/, ghi đường dẫn vào SQL Server."""
    safe_name = name.strip().replace(" ", "_")
    
    # Lưu file tạm trước
    temp_filename = f"temp_{uuid.uuid4().hex[:6]}_{file.filename}"
    temp_path = os.path.join("uploads", temp_filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Chuyển đổi sang wav chuẩn (16kHz, mono) dùng pydub
    from pydub import AudioSegment
    try:
        sound = AudioSegment.from_file(temp_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        
        filename  = f"{safe_name}_{uuid.uuid4().hex[:6]}.wav"
        file_path = os.path.join(db.VOICE_DIR, filename)
        sound.export(file_path, format="wav")
        
        # Xóa file tạm
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        speaker_id = db.register_speaker(name=name, voice_file_path=file_path)
        return {
            "status": "ok",
            "speaker_id": speaker_id,
            "name": name,
            "voice_path": file_path
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"[Register Speaker Error]: {e}")
        return {"status": "error", "message": f"Không thể xử lý âm thanh: {e}"}

@app.get("/speakers")
async def list_speakers():
    """Lấy danh sách người nói đã đăng ký từ SQL Server."""
    return db.get_all_speakers()

# =====================================================================
# ENDPOINT: Giao tiếp, trò chuyện với Trợ lý ảo AI
# =====================================================================
class ChatRequest(BaseModel):
    text: str

@app.post("/chat-assistant")
def api_chat_assistant(req: ChatRequest):
    try:
        reply = llm_proc.chat(req.text)
        return {"status": "ok", "reply": reply}
    except Exception as e:
        print(f"[API Error] /chat-assistant: {e}")
        return {"status": "error", "reply": "Xin lỗi, tôi gặp sự cố kết nối."}

# =====================================================================
# ENDPOINTS: Lịch sử cuộc họp (Sửa lỗi màn hình đen)
# =====================================================================
@app.get("/meetings")
def api_get_meetings():
    try:
        return db.get_all_meetings()
    except Exception as e:
        print(f"[API Error] /meetings: {e}")
        return []

@app.get("/meeting/{meeting_id}/details")
def api_meeting_details(meeting_id: int):
    try:
        # Lấy thông tin cuộc họp và minutes_path từ database
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.id, m.meeting_code, m.title, m.started_at, m.status, m.audio_path, mm.minutes_path
                FROM meetings m
                LEFT JOIN meeting_minutes mm ON m.id = mm.meeting_id
                WHERE m.id = ?
            """, meeting_id)
            row = cursor.fetchone()
            
        if not row:
            return {"status": "error", "message": "Không tìm thấy cuộc họp"}
            
        m_id, meeting_code, title, started_at, status, audio_path, minutes_path = row
        
        # Đọc nội dung file biên bản nếu có
        minutes_content = ""
        if minutes_path and os.path.exists(minutes_path):
            try:
                with open(minutes_path, "r", encoding="utf-8") as f:
                    minutes_content = f.read()
            except Exception as read_err:
                print(f"Error reading minutes file {minutes_path}: {read_err}")
                minutes_content = f"Lỗi đọc file biên bản: {read_err}"
        
        # Lấy transcript
        transcript = db.get_full_transcript(meeting_id)
        
        return {
            "id": m_id,
            "meeting_code": meeting_code,
            "title": title or f"Cuộc họp {meeting_code}",
            "started_at": str(started_at) if started_at else None,
            "status": status,
            "audio_path": audio_path,
            "minutes": minutes_content or "Không có nội dung biên bản.",
            "transcript": transcript
        }
    except Exception as e:
        print(f"[API Error] /meeting/{meeting_id}/details: {e}")
        return {"status": "error", "message": str(e)}

# =====================================================================
# ENDPOINT: Xóa cuộc họp
# =====================================================================
@app.delete("/meeting/{meeting_id}")
def api_delete_meeting(meeting_id: int):
    try:
        success = db.delete_meeting(meeting_id)
        if success:
            return {"status": "ok", "message": f"Đã xóa cuộc họp {meeting_id}"}
        else:
            return {"status": "error", "message": f"Không thể xóa cuộc họp {meeting_id}"}
    except Exception as e:
        print(f"[API Error] DELETE /meeting/{meeting_id}: {e}")
        return {"status": "error", "message": str(e)}

# =====================================================================
# ENDPOINT: Chuyển văn bản thành giọng nói (TTS) bằng giọng đọc Despina
# =====================================================================
class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
def api_tts(req: TTSRequest):
    if not GEMINI_API_KEY:
        return {"status": "error", "message": "Không có GEMINI_API_KEY cấu hình"}
    try:
        import google.generativeai as genai
        from fastapi.responses import Response
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        response = model.generate_content(
            req.text,
            generation_config={
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": "Despina"
                        }
                    }
                }
            }
        )
        
        # Tìm phần dữ liệu audio trong response
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return Response(
                    content=part.inline_data.data, 
                    media_type=part.inline_data.mime_type
                )
                
        return {"status": "error", "message": "Không nhận được dữ liệu âm thanh từ Gemini"}
    except Exception as e:
        print(f"[TTS Error] Gemini Audio generation failed: {e}")
        return {"status": "error", "message": str(e)}

# =====================================================================
# Models for Chrome Extension (Web Speech API mode)
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

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """API dùng cho Frontend Dashboard để phân tích file offline (Diarization + Voice Biometrics)."""
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1] or ".wav"
    file_path = os.path.join("uploads", f"{file_id}{file_ext}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Tạo bản ghi cuộc họp trong SQL Server
    meeting_id = db.create_meeting(meeting_code=file_id)

    # 2. Xử lý âm thanh đồng bộ
    try:
        references = db.get_speaker_voice_paths()
        print(f"Bat dau xu ly am thanh dong bo: {file_path}")
        transcript = audio_proc.process_audio(file_path, references=references if references else None)

        # Lưu từng dòng transcript vào SQL Server
        for turn in transcript:
            db.save_transcript_line(
                meeting_id  = meeting_id,
                text        = turn["text"],
                speaker_name= turn.get("speaker"),
                start_sec   = turn.get("start"),
                end_sec     = turn.get("end")
            )

        # Tạo biên bản
        print("Bat dau tom tat bang LLM...")
        minutes_markdown = llm_proc.generate_minutes(transcript)

        # Lưu biên bản
        minutes_path = db.save_meeting_minutes(meeting_id, minutes_markdown)

        # Save transcript file
        transcript_filename = f"transcript_meeting{meeting_id}_{file_id}.txt"
        transcript_file_path = os.path.join(db.TEXT_DIR, transcript_filename)
        with open(transcript_file_path, "w", encoding="utf-8") as tf:
            for turn in transcript:
                tf.write(f"{turn.get('speaker', 'Unknown')}: {turn['text']}\n")

        # Move/copy audio file to db.VOICE_DIR and update paths
        voice_filename = f"meeting_{meeting_id}_audio_record{file_ext}"
        voice_file_path = os.path.join(db.VOICE_DIR, voice_filename)
        shutil.copy(file_path, voice_file_path)

        db.update_meeting_paths(meeting_id, audio_path=voice_file_path, transcript_path=transcript_file_path)

        from datetime import datetime
        meeting_title = f"Cuộc họp trực tiếp {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        db.end_meeting(meeting_id, title=meeting_title)

        # Lưu JSON kết quả
        result_data = {"file_id": file_id, "transcript": transcript, "minutes": minutes_markdown}
        with open(os.path.join(RESULT_DIR, f"{file_id}.json"), "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

        return {
            "status": "ok",
            "file_id": file_id,
            "meeting_id": meeting_id,
            "transcript": transcript,
            "minutes": minutes_markdown
        }

    except Exception as e:
        print(f"Loi khi xu ly offline: {str(e)}")
        db.end_meeting(meeting_id, title=f"Loi: {str(e)[:200]}")
        return {
            "status": "error",
            "error": str(e),
            "message": f"Lỗi khi xử lý âm thanh: {str(e)}"
        }

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
    # Đồng thời lưu vào room_states cho đồng bộ real-time
    for room_id, rstate in room_states.items():
        if rstate["db_meeting_id"] == req.meeting_id:
            rstate["lines"].append({"speaker": req.speaker, "text": req.text})
            break
    return {"status": "ok"}

@app.post("/finalize-meeting")
def api_finalize_meeting(req: FinalizeMeetingRequest, background_tasks: BackgroundTasks):
    room_states[req.room_id] = {
        "state": "finalizing",
        "db_meeting_id": req.meeting_id,
        "lines": req.transcript
    }
    background_tasks.add_task(finalize_meeting_from_trigger, req.room_id, req.meeting_id)
    return {"status": "finalizing"}

@app.post("/upload-voice")
async def api_upload_voice(meeting_id: int = Form(...), file: UploadFile = File(...)):
    """Lưu trữ file âm thanh của cuộc họp từ Chrome extension."""
    ext = os.path.splitext(file.filename)[1] or ".webm"
    path = os.path.join(db.VOICE_DIR, f"meeting_{meeting_id}_audio_record{ext}")
    
    with open(path, "ab") as f:
        f.write(await file.read())
        
    db.update_meeting_paths(meeting_id, audio_path=path)
    print(f"[DB] Da luu audio chunk vao: {path}")
    return {"status": "ok", "path": path}

@app.get("/result/{file_id}")
async def get_result(file_id: str):
    file_path = os.path.join(RESULT_DIR, f"{file_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "error", "message": "Result not found or still processing."}

# =====================================================================
# REAL-TIME ENDPOINTS cho Chrome Extension
# =====================================================================

@app.post("/upload-chunk")
async def upload_chunk(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    room_id: str = Form(...)          # Mã phòng Google Meet (VD: abc-defg-hij)
):
    """
    Nhận chunk audio 5s từ Chrome Extension.
    - Standby  : Chay Viettel STT, kiem tra trigger "bat dau cuoc hop"
    - Recording: Chay STT + luu log vao DB
    - Finalizing: Da dang xu ly, bo qua chunk moi
    """
    # Khoi tao state neu phong chua co
    if room_id not in room_states:
        room_states[room_id] = {
            "state": "standby",
            "db_meeting_id": None,
            "lines": []
        }

    state = room_states[room_id]["state"]

    # Neu dang xu ly cuoi cuoc hop, bo qua chunk moi
    if state == "finalizing":
        return {"status": "finalizing", "room_id": room_id}

    # Luu chunk tam
    chunk_path = os.path.join("uploads", f"{room_id}_chunk_{uuid.uuid4().hex[:6]}.webm")
    with open(chunk_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(process_chunk_with_trigger, chunk_path, room_id)

    return {
        "status": "ok",
        "room_state": state,
        "room_id": room_id
    }


async def process_chunk_with_trigger(chunk_path: str, room_id: str):
    """
    Xu ly chunk audio:
    - Standby  : Chi doc text, kiem tra xem co tu khoa 'bat dau cuoc hop' khong
    - Recording: Doc text + luu DB + kiem tra 'ket thuc cuoc hop'
    """
    try:
        # STT - Chuyen am thanh thanh van ban
        text = audio_proc.stt_viettel(chunk_path)
        if os.path.exists(chunk_path):
            os.remove(chunk_path)

        if not text or not text.strip():
            return

        text = text.strip()
        current_state = room_states[room_id]["state"]

        # -------------------------------------------------------
        # STANDBY: Doi lenh "bat dau cuoc hop"
        # -------------------------------------------------------
        if current_state == "standby":
            print(f"[{room_id}][STANDBY] Nghe thay: '{text[:60]}'")

            if check_start_trigger(text):
                # Chuyen sang RECORDING
                meeting_id_db = db.create_meeting(meeting_code=room_id)
                room_states[room_id]["state"]         = "recording"
                room_states[room_id]["db_meeting_id"] = meeting_id_db
                room_states[room_id]["lines"]         = []
                print(f"[{room_id}] >>> BAT DAU GHI BIEN BAN! Meeting DB ID: {meeting_id_db}")

        # -------------------------------------------------------
        # RECORDING: Ghi log moi thu, doi lenh "ket thuc cuoc hop"
        # -------------------------------------------------------
        elif current_state == "recording":
            meeting_id_db = room_states[room_id]["db_meeting_id"]
            print(f"[{room_id}][RECORDING] '{text[:60]}'")

            # Luu dong hoi thoai vao logs/text + SQL
            db.save_transcript_line(
                meeting_id   = meeting_id_db,
                text         = text,
                speaker_name = "Speaker",   # Diarization se nhan dien sau
                start_sec    = None,
                end_sec      = None
            )
            room_states[room_id]["lines"].append({
                "speaker": "Speaker",
                "text": text
            })

            # Kiem tra lenh ket thuc
            if check_stop_trigger(text):
                room_states[room_id]["state"] = "finalizing"
                print(f"[{room_id}] >>> KET THUC CUOC HOP! Dang tao bien ban...")

                # Tao bien ban bang Gemini trong nen
                import asyncio
                asyncio.create_task(
                    finalize_meeting_from_trigger(room_id, meeting_id_db)
                )

    except Exception as e:
        print(f"[{room_id}] Loi process_chunk: {e}")


async def finalize_meeting_from_trigger(room_id: str, meeting_id_db: int):
    """Duoc goi khi nghe thay 'ket thuc cuoc hop'. Tao bien ban va luu DB."""
    try:
        transcript = room_states[room_id]["lines"]
        if not transcript:
            print(f"[{room_id}] Khong co noi dung de tao bien ban.")
            return

        # Goi Gemini tao bien ban
        minutes_md = llm_proc.generate_minutes(transcript)

        # Luu bien ban vao logs/minutes + SQL
        minutes_path = db.save_meeting_minutes(meeting_id_db, minutes_md)
        db.end_meeting(meeting_id_db)

        # Luu JSON cho Frontend hien thi
        result = {
            "file_id": room_id,
            "meeting_db_id": meeting_id_db,
            "transcript": transcript,
            "minutes": minutes_md,
            "minutes_file": minutes_path
        }
        with open(os.path.join(RESULT_DIR, f"{room_id}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        print(f"[{room_id}] Bien ban da luu: {minutes_path}")

    except Exception as e:
        print(f"[{room_id}] Loi finalize: {e}")
    finally:
        # Reset trang thai ve standby cho cuoc hop tiep theo
        room_states[room_id] = {"state": "standby", "db_meeting_id": None, "lines": []}



@app.get("/live-transcript/{meeting_id}")
async def get_live_transcript(meeting_id: str):
    """Trả về transcript real-time cho Extension popup."""
    if meeting_id not in live_sessions:
        return {"lines": [], "status": "waiting"}
    return {
        "lines": live_sessions[meeting_id]["lines"],
        "status": "recording"
    }

@app.post("/end-meeting/{meeting_id}")
async def end_meeting(meeting_id: str, background_tasks: BackgroundTasks):
    """Kết thúc cuộc họp: chạy Gemini để tạo biên bản hoàn chỉnh từ toàn bộ transcript."""
    if meeting_id not in live_sessions:
        return {"status": "error", "message": "Không tìm thấy phiên họp."}
    
    transcript = live_sessions[meeting_id]["lines"]
    background_tasks.add_task(finalize_meeting, transcript, meeting_id)
    
    return {"status": "finalizing", "meeting_id": meeting_id}

async def finalize_meeting(transcript: list, meeting_id: str):
    """Dùng Gemini để tóm tắt toàn bộ biên bản sau cuộc họp."""
    try:
        minutes_markdown = llm_proc.generate_minutes(transcript)
        result_data = {
            "file_id": meeting_id,
            "transcript": transcript,
            "minutes": minutes_markdown
        }
        with open(os.path.join(RESULT_DIR, f"{meeting_id}.json"), "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
        print(f"Bien ban cuoc hop {meeting_id} da hoan tat.")
        # Dọn dẹp session
        del live_sessions[meeting_id]
    except Exception as e:
        print(f"Loi finalize meeting: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
