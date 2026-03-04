from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import json
from dotenv import load_dotenv
from typing import Optional

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def process_meeting(file_path: str, file_id: str, meeting_id: int = None):
    try:
        # Lấy giọng mẫu từ SQL Server để nhận diện tên người nói
        references = db.get_speaker_voice_paths()

        # Bước 1: Speaker Diarization + Viettel STT
        print(f"Bắt đầu xử lý âm thanh: {file_path}")
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
        print("Bắt đầu tóm tắt bằng LLM...")
        minutes_markdown = llm_proc.generate_minutes(transcript)

        # Bước 4: Lưu biên bản vào logs/minutes + SQL Server
        minutes_path = db.save_meeting_minutes(meeting_id, minutes_markdown) if meeting_id else None
        if meeting_id:
            db.end_meeting(meeting_id)

        # Bước 5: Lưu JSON kết quả cho Frontend
        result_data = {"file_id": file_id, "transcript": transcript, "minutes": minutes_markdown}
        with open(os.path.join(RESULT_DIR, f"{file_id}.json"), "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

        print(f"Xử lý hoàn tất cho ID: {file_id}")
    except Exception as e:
        print(f"Lỗi khi xử lý: {str(e)}")

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
    """Nhận file giọng mẫu, lưu vào logs/voice/, ghi đường dẫn vào SQL Server."""
    safe_name = name.strip().replace(" ", "_")
    file_ext  = os.path.splitext(file.filename)[1] or ".wav"
    filename  = f"{safe_name}_{uuid.uuid4().hex[:6]}{file_ext}"
    file_path = os.path.join(db.VOICE_DIR, filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    speaker_id = db.register_speaker(name=name, voice_file_path=file_path)
    return {
        "status": "ok",
        "speaker_id": speaker_id,
        "name": name,
        "voice_path": file_path
    }

@app.get("/speakers")
async def list_speakers():
    """Lấy danh sách người nói đã đăng ký từ SQL Server."""
    return db.get_all_speakers()

# =====================================================================
@app.post("/upload-audio")
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    file_path = os.path.join("uploads", f"{file_id}{file_ext}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Tạo bản ghi cuộc họp trong SQL Server
    meeting_id = db.create_meeting(meeting_code=file_id)

    # Chạy quy trình AI trong nền
    background_tasks.add_task(process_meeting, file_path, file_id, meeting_id)

    return {
        "status": "processing",
        "file_id": file_id,
        "meeting_id": meeting_id,
        "message": "Đang xử lý âm thanh trong nền."
    }

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
        print(f"Biên bản cuộc họp {meeting_id} đã hoàn tất.")
        # Dọn dẹp session
        del live_sessions[meeting_id]
    except Exception as e:
        print(f"Lỗi finalize meeting: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
