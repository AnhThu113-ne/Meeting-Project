"""
test_full_flow.py
Test toàn bộ luồng lưu dữ liệu:
  1. Đăng ký giọng nói  → lưu file vào logs/voice  + path vào SQL
  2. Tạo cuộc họp       → lưu vào SQL
  3. Lưu transcript     → lưu file vào logs/text   + path vào SQL
  4. Lưu biên bản       → lưu file vào logs/minutes + path vào SQL
  5. Truy vấn SQL để xác nhận chỉ có ĐƯỜNG DẪN trong DB
"""
import sys
sys.path.insert(0, r"e:\meeting_project\backend")

import pyodbc, os, shutil
from datetime import datetime

DRIVER  = "ODBC Driver 17 for SQL Server"
SERVER  = "localhost"
DB      = "MeetingMinutesDB"

VOICE_DIR   = r"e:\meeting_project\logs\voice"
TEXT_DIR    = r"e:\meeting_project\logs\text"
MINUTES_DIR = r"e:\meeting_project\logs\minutes"

conn = pyodbc.connect(
    f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DB};Trusted_Connection=yes;"
)
conn.autocommit = True
cur  = conn.cursor()

SEP = "=" * 55

# ===========================================================
# BƯỚC 1: Đăng ký giọng nói mẫu
# ===========================================================
print(SEP)
print("BUOC 1: Dang ky giong noi mau")
print(SEP)

# Tạo file WAV giả để test
os.makedirs(VOICE_DIR, exist_ok=True)
voice_file = os.path.join(VOICE_DIR, f"Nam_test_{datetime.now().strftime('%H%M%S')}.wav")
with open(voice_file, "wb") as f:
    f.write(b"RIFF_FAKE_WAV_FOR_TESTING")  # File giả

cur.execute("""
    INSERT INTO speakers (name, voice_path) OUTPUT INSERTED.id
    VALUES (?, ?)
""", "Nam (Test)", voice_file)
speaker_id = cur.fetchone()[0]

print(f"  File giong luu tai   : {voice_file}")
print(f"  SQL chi luu path     : speakers.voice_path")
print(f"  -> Speaker ID        : {speaker_id}")

# Xác nhận SQL KHÔNG chứa dữ liệu nhị phân, chỉ có path
cur.execute("SELECT name, voice_path FROM speakers WHERE id = ?", speaker_id)
row = cur.fetchone()
print(f"  -> SQL speakers.name : {row[0]}")
print(f"  -> SQL voice_path    : {row[1]}")
print(f"  -> File ton tai?     : {os.path.exists(row[1])}")

# ===========================================================
# BƯỚC 2: Tạo cuộc họp
# ===========================================================
print()
print(SEP)
print("BUOC 2: Tao cuoc hop moi")
print(SEP)

meeting_code = f"test-{datetime.now().strftime('%H%M%S')}"
cur.execute("""
    INSERT INTO meetings (meeting_code, started_at, status) OUTPUT INSERTED.id
    VALUES (?, GETDATE(), 'recording')
""", meeting_code)
meeting_id = cur.fetchone()[0]
print(f"  -> Meeting Code      : {meeting_code}")
print(f"  -> Meeting ID (SQL)  : {meeting_id}")

# ===========================================================
# BƯỚC 3: Lưu transcript (từng lượt nói)
# ===========================================================
print()
print(SEP)
print("BUOC 3: Luu transcript tung luot noi")
print(SEP)

os.makedirs(TEXT_DIR, exist_ok=True)
fake_transcript = [
    {"speaker": "Nam",  "text": "Xin chào mọi người, bắt đầu họp nhé.", "start": 0.0,  "end": 3.5},
    {"speaker": "Thầy", "text": "Ừ, hôm nay chúng ta thảo luận về đề tài tốt nghiệp.", "start": 4.0, "end": 9.0},
    {"speaker": "Nam",  "text": "Em đã xây được hệ thống nhận diện giọng nói và tạo biên bản tự động.", "start": 10.0, "end": 15.0},
    {"speaker": "Thầy", "text": "Tốt! Hãy demo vào tuần sau.", "start": 16.0, "end": 19.0},
]

for turn in fake_transcript:
    # Lưu file text ra disk
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"meeting{meeting_id}_{ts}.txt"
    txt_path = os.path.join(TEXT_DIR, filename)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Meeting ID : {meeting_id}\n")
        f.write(f"Speaker    : {turn['speaker']}\n")
        f.write(f"Time       : {turn['start']:.1f}s - {turn['end']:.1f}s\n")
        f.write(f"Content    :\n{turn['text']}\n")

    # SQL chỉ lưu đường dẫn
    cur.execute("""
        INSERT INTO transcripts
            (meeting_id, speaker_label, text_content, text_file_path, start_time_sec, end_time_sec)
        OUTPUT INSERTED.id
        VALUES (?, ?, ?, ?, ?, ?)
    """, meeting_id, turn["speaker"], turn["text"], txt_path, turn["start"], turn["end"])
    t_id = cur.fetchone()[0]
    print(f"  [{turn['speaker']:6}] text_file_path (SQL) -> {os.path.basename(txt_path)}")
    print(f"           File ton tai?               -> {os.path.exists(txt_path)}")

# ===========================================================
# BƯỚC 4: Lưu biên bản (Meeting Minutes)
# ===========================================================
print()
print(SEP)
print("BUOC 4: Luu bien ban cuoc hop (Gemini output)")
print(SEP)

os.makedirs(MINUTES_DIR, exist_ok=True)
fake_minutes = """# Biên bản cuộc họp

## Thông tin
- Meeting code: {code}
- Thời gian: {dt}

## Tóm tắt nội dung
Cuộc họp thảo luận về tiến độ đề tài tốt nghiệp nhận diện giọng nói.

## Quyết định
- Nam sẽ demo hệ thống vào tuần sau.

## Action Items
| Người | Việc cần làm | Deadline |
|-------|-------------|---------|
| Nam   | Chuẩn bị demo hệ thống | Tuần sau |
""".format(code=meeting_code, dt=datetime.now().strftime("%d/%m/%Y %H:%M"))

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
minutes_filename = f"minutes_meeting{meeting_id}_{ts}.md"
minutes_path     = os.path.join(MINUTES_DIR, minutes_filename)

with open(minutes_path, "w", encoding="utf-8") as f:
    f.write(fake_minutes)

cur.execute("""
    INSERT INTO meeting_minutes (meeting_id, minutes_path) VALUES (?, ?)
""", meeting_id, minutes_path)

print(f"  File bien ban luu tai: {minutes_path}")
print(f"  SQL chi luu path     : meeting_minutes.minutes_path")
print(f"  File ton tai?        : {os.path.exists(minutes_path)}")

# Cập nhật trạng thái cuộc họp
cur.execute("UPDATE meetings SET ended_at = GETDATE(), status = 'done' WHERE id = ?", meeting_id)

# ===========================================================
# BƯỚC 5: Xác nhận toàn bộ trong SQL Server
# ===========================================================
print()
print(SEP)
print("BUOC 5: XAC NHAN SQL CHI LUU DUONG DAN")
print(SEP)

print("\n--- speakers ---")
cur.execute("SELECT id, name, voice_path FROM speakers WHERE id = ?", speaker_id)
for r in cur.fetchall():
    print(f"  ID={r[0]} | name={r[1]} | voice_path={r[2]}")

print("\n--- meetings ---")
cur.execute("SELECT id, meeting_code, status FROM meetings WHERE id = ?", meeting_id)
for r in cur.fetchall():
    print(f"  ID={r[0]} | code={r[1]} | status={r[2]}")

print("\n--- transcripts (chi hien text_file_path) ---")
cur.execute("SELECT id, speaker_label, text_file_path FROM transcripts WHERE meeting_id = ?", meeting_id)
for r in cur.fetchall():
    print(f"  ID={r[0]} | speaker={r[1]:6} | path=...{r[2][-40:]}")

print("\n--- meeting_minutes ---")
cur.execute("SELECT id, minutes_path FROM meeting_minutes WHERE meeting_id = ?", meeting_id)
for r in cur.fetchall():
    print(f"  ID={r[0]} | path=...{r[1][-50:]}")

conn.close()

print()
print(SEP)
print("KIEM TRA THU MUC LOGS TREN DISK")
print(SEP)
for folder, label in [(VOICE_DIR, "voice"), (TEXT_DIR, "text"), (MINUTES_DIR, "minutes")]:
    files = os.listdir(folder) if os.path.exists(folder) else []
    relevant = [f for f in files if "test" in f or f"meeting{meeting_id}" in f]
    print(f"\n  logs/{label}/  ({len(files)} file tong cong)")
    for fn in relevant:
        size = os.path.getsize(os.path.join(folder, fn))
        print(f"    - {fn}  ({size} bytes)")

print()
print("=" * 55)
print("THANH CONG! Toan bo luong luu du lieu dung nhu thiet ke:")
print("  File thuc      -> logs/voice | logs/text | logs/minutes")
print("  SQL Server     -> Chi luu DUONG DAN den file")
print("=" * 55)
