"""
database.py
Quản lý kết nối và thao tác với SQL Server cho dự án Meeting Minutes.
Dùng pyodbc để kết nối SQL Server.
"""
import pyodbc
import os
import sys
from datetime import datetime

# Reconfigure console stream to UTF-8 to prevent encoding crashes on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


# ==============================================================
# CẤU HÌNH KẾT NỐI SQL SERVER
# Chỉnh SERVER và DATABASE theo máy của bạn
# ==============================================================
# Cấu hình đường dẫn thư mục lưu file tương đối so với thư mục gốc dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR   = os.path.join(BASE_DIR, "logs", "voice")
TEXT_DIR    = os.path.join(BASE_DIR, "logs", "text")
MINUTES_DIR = os.path.join(BASE_DIR, "logs", "minutes")

for d in [VOICE_DIR, TEXT_DIR, MINUTES_DIR]:
    os.makedirs(d, exist_ok=True)

_cached_connection_string = None

def get_connection():
    """Mở kết nối đến SQL Server với khả năng tự động dò tìm server phù hợp."""
    global _cached_connection_string
    if _cached_connection_string:
        return pyodbc.connect(_cached_connection_string)

    SQL_SERVER   = os.getenv("SQL_SERVER")
    SQL_DATABASE = os.getenv("SQL_DATABASE", "MeetingMinutesDB")
    DRIVER       = "ODBC Driver 17 for SQL Server"

    # Danh sách các server sẽ thử kết nối
    servers_to_try = []
    if SQL_SERVER:
        servers_to_try.append(SQL_SERVER)
    
    # Thử các named instances phổ biến và localhost
    servers_to_try.extend([
        r"localhost\ATHU2019",
        r"localhost\SQLEXPRESS",
        r"localhost",
        r"localhost\THU2019",
        r"localhost\SQLEXPRESS01"
    ])

    last_error = None
    for server in servers_to_try:
        conn_str = f"DRIVER={{{DRIVER}}};SERVER={server};DATABASE={SQL_DATABASE};Trusted_Connection=yes;"
        try:
            conn = pyodbc.connect(conn_str, timeout=3)
            print(f"[DB] Ket noi thanh cong toi SQL Server: {server}")
            _cached_connection_string = conn_str
            return conn
        except Exception as e:
            last_error = e
            continue

    if last_error:
        print(f"[DB] Loi: Khong the ket noi toi bat ky SQL Server instance nao: {servers_to_try}")
        raise last_error



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
    print(f"[DB] Da dang ky giong noi cho '{name}' | ID: {speaker_id} | Path: {voice_file_path}")
    return speaker_id


def get_all_speakers():
    """Lấy danh sách tất cả người nói đã đăng ký."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, voice_path, created_at FROM speakers WHERE is_active = 1")
        rows = cursor.fetchall()
    return [{"id": r[0], "name": r[1], "voice_path": r[2], "created_at": str(r[3]) if r[3] else None} for r in rows]


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
    print(f"[DB] Cuoc hop moi: {meeting_code} | ID: {meeting_id}")
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
    print(f"[DB] Cuoc hop ID={meeting_id} da ket thuc.")

def update_meeting_paths(meeting_id: int, audio_path: str = None, transcript_path: str = None):
    """Lưu đường dẫn audio/transcript tổng hợp vào bảng meetings."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if audio_path:
            cursor.execute("UPDATE meetings SET audio_path = ? WHERE id = ?", audio_path, meeting_id)
        if transcript_path:
            cursor.execute("UPDATE meetings SET transcript_path = ? WHERE id = ?", transcript_path, meeting_id)
        conn.commit()
    print(f"[DB] Da cap nhat duong dan cho cuoc hop ID={meeting_id}")


def get_meeting_by_code(meeting_code: str):
    """Lấy thông tin cuộc họp theo meeting_code."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, meeting_code, title, started_at, status, audio_path, transcript_path FROM meetings WHERE meeting_code = ?", meeting_code)
        row = cursor.fetchone()
    if row:
        return {
            "id": row[0], "meeting_code": row[1], "title": row[2], 
            "started_at": str(row[3]) if row[3] else None, "status": row[4],
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
        "started_at": str(r[3]) if r[3] else None, "status": r[4], 
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

    print(f"[DB] Transcript luu: {file_path}")
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

    print(f"[DB] Bien ban da luu: {file_path}")
    return file_path


def delete_meeting(meeting_id: int):
    """Xóa cuộc họp khỏi database và xóa các tệp vật lý liên quan trên đĩa."""
    files_to_delete = []
    
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Lấy audio_path của cuộc họp
            cursor.execute("SELECT audio_path FROM meetings WHERE id = ?", meeting_id)
            row = cursor.fetchone()
            if row and row[0]:
                files_to_delete.append(row[0])
                
            # Lấy các file transcript text_file_path
            cursor.execute("SELECT text_file_path FROM transcripts WHERE meeting_id = ?", meeting_id)
            for r in cursor.fetchall():
                if r[0]:
                    files_to_delete.append(r[0])
                    
            # Lấy minutes_path
            cursor.execute("SELECT minutes_path FROM meeting_minutes WHERE meeting_id = ?", meeting_id)
            for r in cursor.fetchall():
                if r[0]:
                    files_to_delete.append(r[0])
                    
            # Xóa các bản ghi trong cơ sở dữ liệu
            cursor.execute("DELETE FROM transcripts WHERE meeting_id = ?", meeting_id)
            cursor.execute("DELETE FROM meeting_minutes WHERE meeting_id = ?", meeting_id)
            cursor.execute("DELETE FROM meetings WHERE id = ?", meeting_id)
            conn.commit()
            
        # Xóa các tệp vật lý trên đĩa
        for fpath in files_to_delete:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    print(f"[DB] Da xoa file vat ly: {fpath}")
            except Exception as file_err:
                print(f"[DB Error] Khong the xoa file {fpath}: {file_err}")
                
        print(f"[DB] Da xoa hoan toan cuoc hop ID={meeting_id}")
        return True
    except Exception as e:
        print(f"[DB Error] Loi khi xoa cuoc hop ID={meeting_id}: {e}")
        return False
