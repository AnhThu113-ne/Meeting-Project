import pyodbc

DRIVER = "ODBC Driver 17 for SQL Server"
SERVER = "localhost"

print("=== Ket noi SQL Server ===")

# Buoc 1: Tao database
conn = pyodbc.connect(
    f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE=master;Trusted_Connection=yes;"
)
conn.autocommit = True
cursor = conn.cursor()
cursor.execute("""
    IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'MeetingMinutesDB')
    BEGIN
        CREATE DATABASE MeetingMinutesDB
        PRINT 'Da tao MeetingMinutesDB'
    END
    ELSE
        PRINT 'MeetingMinutesDB da ton tai'
""")
print("[OK] Database MeetingMinutesDB san sang")
conn.close()

# Buoc 2: Tao cac bang
conn = pyodbc.connect(
    f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE=MeetingMinutesDB;Trusted_Connection=yes;"
)
conn.autocommit = True
cursor = conn.cursor()

cursor.execute("""
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='speakers' AND xtype='U')
CREATE TABLE speakers (
    id          INT IDENTITY(1,1) PRIMARY KEY,
    name        NVARCHAR(100)   NOT NULL,
    voice_path  NVARCHAR(500)   NOT NULL,
    created_at  DATETIME2       DEFAULT GETDATE(),
    is_active   BIT             DEFAULT 1
)
""")
print("[OK] Bang speakers")

cursor.execute("""
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='meetings' AND xtype='U')
CREATE TABLE meetings (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    meeting_code    NVARCHAR(100)   NOT NULL UNIQUE,
    title           NVARCHAR(300)   NULL,
    started_at      DATETIME2       NULL,
    ended_at        DATETIME2       NULL,
    created_at      DATETIME2       DEFAULT GETDATE(),
    status          NVARCHAR(50)    DEFAULT 'recording'
)
""")
print("[OK] Bang meetings")

cursor.execute("""
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='transcripts' AND xtype='U')
CREATE TABLE transcripts (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    meeting_id      INT             NOT NULL,
    speaker_id      INT             NULL,
    speaker_label   NVARCHAR(100)   NULL,
    text_content    NVARCHAR(MAX)   NOT NULL,
    text_file_path  NVARCHAR(500)   NULL,
    start_time_sec  FLOAT           NULL,
    end_time_sec    FLOAT           NULL,
    created_at      DATETIME2       DEFAULT GETDATE(),
    FOREIGN KEY (meeting_id) REFERENCES meetings(id)
)
""")
print("[OK] Bang transcripts")

cursor.execute("""
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='meeting_minutes' AND xtype='U')
CREATE TABLE meeting_minutes (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    meeting_id      INT             NOT NULL,
    minutes_path    NVARCHAR(500)   NOT NULL,
    created_at      DATETIME2       DEFAULT GETDATE(),
    FOREIGN KEY (meeting_id) REFERENCES meetings(id)
)
""")
print("[OK] Bang meeting_minutes")

# Kiem tra ket qua
cursor.execute("""
    SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'BASE TABLE'
""")
tables = [row[0] for row in cursor.fetchall()]
print(f"\n=== Cac bang trong MeetingMinutesDB ===")
for t in tables:
    print(f"  - {t}")

conn.close()
print("\n[THANH CONG] SQL Server da ket noi va khoi tao xong!")
