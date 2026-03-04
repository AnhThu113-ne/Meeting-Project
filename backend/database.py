"""
database.py
Quản lý kết nối và thao tác với SQL Server cho dự án Meeting Minutes.
Dùng pyodbc để kết nối SQL Server.
"""
import pyodbc
import os
from datetime import datetime

# ==============================================================
# CẤU HÌNH KẾT NỐI SQL SERVER
# Chỉnh SERVER và DATABASE theo máy của bạn
# ==============================================================
SQL_SERVER   = os.getenv("SQL_SERVER", "localhost")      # Hoặc DESKTOP-XXX\SQLEXPRESS
SQL_DATABASE = os.getenv("SQL_DATABASE", "MeetingMinutesDB")

# Dùng Windows Authentication (không cần username/password)
CONNECTION_STRING = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SQL_SERVER};"
    f"DATABASE={SQL_DATABASE};"
    f"Trusted_Connection=yes;"
)

# Thư mục lưu file trên disk
VOICE_DIR   = r"e:\meeting_project\logs\voice"
TEXT_DIR    = r"e:\meeting_project\logs\text"
MINUTES_DIR = r"e:\meeting_project\logs\minutes"

for d in [VOICE_DIR, TEXT_DIR, MINUTES_DIR]:
    os.makedirs(d, exist_ok=True)


def get_connection():
    """Mở kết nối đến SQL Server."""
    return pyodbc.connect(CONNECTION_STRING)


# ==============================================================
# SPEAKERS
# ==============================================================

def register_speaker(name: str, voice_file_path: str) -> int:
    """
    Đăng ký người nói mới.
    - name: Tên người (VD: 'Nam', 'Thầy')
    - voice_file_path: Đường dẫn file .wav mẫu đã lưu trong VOICE_DIR
    - Trả về: speaker_id
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO speakers (name, voice_path, created_at)
            OUTPUT INSERTED.id
            VALUES (?, ?, GETDATE())
        """, name, voice_file_path)
        speaker_id = cursor.fetchone()[0]
        conn.commit()
    print(f"[DB] Đã đăng ký giọng nói cho '{name}' | ID: {speaker_id} | Path: {voice_file_path}")
    return speaker_id


def get_all_speakers():
    """Lấy danh sách tất cả người nói đã đăng ký."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, voice_path, created_at FROM speakers WHERE is_active = 1")
        rows = cursor.fetchall()
    return [{"id": r[0], "name": r[1], "voice_path": r[2], "created_at": str(r[3])} for r in rows]


def get_speaker_voice_paths() -> dict:
    """Trả về dict {name: voice_path} để dùng với identify_speaker()."""
    speakers = get_all_speakers()
    return {s["name"]: s["voice_path"] for s in speakers}


# ==============================================================
# MEETINGS
# ==============================================================

def create_meeting(meeting_code: str) -> int:
    """
    Tạo bản ghi cuộc họp mới khi bắt đầu ghi âm.
    - Trả về: meeting_id
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO meetings (meeting_code, started_at, status)
            OUTPUT INSERTED.id
            VALUES (?, GETDATE(), 'recording')
        """, meeting_code)
        meeting_id = cursor.fetchone()[0]
        conn.commit()
    print(f"[DB] Cuộc họp mới: {meeting_code} | ID: {meeting_id}")
    return meeting_id


def end_meeting(meeting_id: int, title: str = None):
    """Cập nhật trạng thái kết thúc cuộc họp."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE meetings
            SET ended_at = GETDATE(), status = 'done', title = ?
            WHERE id = ?
        """, title, meeting_id)
        conn.commit()
    print(f"[DB] Cuộc họp ID={meeting_id} đã kết thúc.")

def update_meeting_paths(meeting_id: int, audio_path: str = None, transcript_path: str = None):
    """Lưu đường dẫn audio/transcript tổng hợp vào bảng meetings."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if audio_path:
            cursor.execute("UPDATE meetings SET audio_path = ? WHERE id = ?", audio_path, meeting_id)
        if transcript_path:
            cursor.execute("UPDATE meetings SET transcript_path = ? WHERE id = ?", transcript_path, meeting_id)
        conn.commit()
    print(f"[DB] Đã cập nhật đường dẫn cho cuộc họp ID={meeting_id}")


def get_meeting_by_code(meeting_code: str):
    """Lấy thông tin cuộc họp theo meeting_code."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, meeting_code, title, started_at, status, audio_path, transcript_path FROM meetings WHERE meeting_code = ?", meeting_code)
        row = cursor.fetchone()
    if row:
        return {
            "id": row[0], "meeting_code": row[1], "title": row[2], 
            "started_at": str(row[3]), "status": row[4],
            "audio_path": row[5], "transcript_path": row[6]
        }
    return None

def get_all_meetings():
    """Lấy danh sách tất cả các cuộc họp, kèm theo thông tin biên bản nếu có."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT m.id, m.meeting_code, m.title, m.started_at, m.status, mm.minutes_path, m.audio_path, m.transcript_path
            FROM meetings m
            LEFT JOIN meeting_minutes mm ON m.id = mm.meeting_id
            ORDER BY m.started_at DESC
        """)
        rows = cursor.fetchall()
    return [{
        "id": r[0], "meeting_code": r[1], "title": r[2], 
        "started_at": str(r[3]), "status": r[4], 
        "has_minutes": bool(r[5]), "minutes_path": r[5],
        "audio_path": r[6], "transcript_path": r[7]
    } for r in rows]

# ==============================================================
# TRANSCRIPTS
# ==============================================================

def save_transcript_line(
    meeting_id: int,
    text: str,
    speaker_name: str = None,
    speaker_id: int = None,
    start_sec: float = None,
    end_sec: float = None
) -> int:
    """
    Lưu một dòng hội thoại:
    1. Ghi nội dung text ra file .txt trong TEXT_DIR
    2. Chỉ lưu đường dẫn vào SQL Server
    - Trả về: transcript_id
    """
    # 1. Lưu text ra file disk
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"meeting{meeting_id}_{timestamp}.txt"
    file_path = os.path.join(TEXT_DIR, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Meeting ID : {meeting_id}\n")
        f.write(f"Speaker    : {speaker_name or 'Unknown'}\n")
        f.write(f"Time       : {start_sec:.1f}s - {end_sec:.1f}s\n" if start_sec else "")
        f.write(f"Content    :\n{text}\n")

    # 2. Lưu đường dẫn vào SQL Server
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transcripts
                (meeting_id, speaker_id, speaker_label, text_content, text_file_path, start_time_sec, end_time_sec)
            OUTPUT INSERTED.id
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, meeting_id, speaker_id, speaker_name, text, file_path, start_sec, end_sec)
        transcript_id = cursor.fetchone()[0]
        conn.commit()

    print(f"[DB] Transcript lưu: {file_path}")
    return transcript_id


def get_full_transcript(meeting_id: int) -> list:
    """Lấy toàn bộ transcript của một cuộc họp."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COALESCE(s.name, t.speaker_label, 'Unknown') AS speaker,
                t.text_content,
                t.text_file_path,
                t.start_time_sec,
                t.end_time_sec
            FROM transcripts t
            LEFT JOIN speakers s ON t.speaker_id = s.id
            WHERE t.meeting_id = ?
            ORDER BY t.start_time_sec, t.id
        """, meeting_id)
        rows = cursor.fetchall()
    return [{"speaker": r[0], "text": r[1], "text_file_path": r[2],
             "start": r[3], "end": r[4]} for r in rows]


# ==============================================================
# MEETING MINUTES (Biên bản)
# ==============================================================

def save_meeting_minutes(meeting_id: int, minutes_text: str) -> str:
    """
    Lưu biên bản cuộc họp:
    1. Ghi ra file .md trong MINUTES_DIR
    2. Chỉ lưu đường dẫn vào SQL Server
    - Trả về: đường dẫn file biên bản
    """
    # 1. Ghi file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"minutes_meeting{meeting_id}_{timestamp}.md"
    file_path = os.path.join(MINUTES_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(minutes_text)

    # 2. Lưu đường dẫn vào SQL
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO meeting_minutes (meeting_id, minutes_path)
            VALUES (?, ?)
        """, meeting_id, file_path)
        conn.commit()

    print(f"[DB] Biên bản đã lưu: {file_path}")
    return file_path
