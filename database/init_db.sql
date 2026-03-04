-- ============================================================
-- DATABASE: MeetingMinutesDB
-- Mô tả: Quản lý giọng nói mẫu, cuộc họp, transcript và biên bản
-- SQL Server
-- ============================================================

-- Tạo database (bỏ qua nếu đã có)
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'MeetingMinutesDB')
BEGIN
    CREATE DATABASE MeetingMinutesDB;
END
GO

USE MeetingMinutesDB;
GO

-- ============================================================
-- BẢNG 1: speakers
-- Lưu thông tin người nói + đường dẫn file giọng mẫu đã đăng ký
-- File giọng thực sự lưu tại: e:\meeting_project\logs\voice\
-- ============================================================
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='speakers' AND xtype='U')
CREATE TABLE speakers (
    id          INT IDENTITY(1,1) PRIMARY KEY,
    name        NVARCHAR(100)   NOT NULL,               -- Tên người nói (VD: "Nam", "Thầy")
    voice_path  NVARCHAR(500)   NOT NULL,               -- Đường dẫn file .wav mẫu
    created_at  DATETIME2       DEFAULT GETDATE(),
    is_active   BIT             DEFAULT 1               -- Có đang được dùng không
);
GO

-- ============================================================
-- BẢNG 2: meetings
-- Lưu thông tin từng cuộc họp
-- ============================================================
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='meetings' AND xtype='U')
CREATE TABLE meetings (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    meeting_code    NVARCHAR(100)   NOT NULL UNIQUE,    -- Mã phòng Meet (VD: abc-defg-hij)
    title           NVARCHAR(300)   NULL,               -- Tiêu đề cuộc họp (do AI tóm tắt)
    started_at      DATETIME2       NULL,
    ended_at        DATETIME2       NULL,
    created_at      DATETIME2       DEFAULT GETDATE(),
    status          NVARCHAR(50)    DEFAULT 'recording', -- recording | processing | done
    audio_path      NVARCHAR(500)   NULL,               -- Đường dẫn file ghi âm tổng
    transcript_path NVARCHAR(500)   NULL                -- Đường dẫn file text tổng hợp
);
GO

-- ============================================================
-- BẢNG 3: transcripts
-- Lưu từng đoạn hội thoại theo mốc thời gian
-- File text thực sự lưu tại: e:\meeting_project\logs\text\
-- ============================================================
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='transcripts' AND xtype='U')
CREATE TABLE transcripts (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    meeting_id      INT             NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    speaker_id      INT             NULL REFERENCES speakers(id),    -- NULL nếu chưa nhận diện được
    speaker_label   NVARCHAR(100)   NULL,               -- Nhãn tạm (SPEAKER_00, SPEAKER_01)
    text_content    NVARCHAR(MAX)   NOT NULL,            -- Nội dung đoạn hội thoại
    text_file_path  NVARCHAR(500)   NULL,               -- Đường dẫn file .txt lưu đoạn này
    start_time_sec  FLOAT           NULL,               -- Thời điểm bắt đầu nói (giây)
    end_time_sec    FLOAT           NULL,               -- Thời điểm kết thúc nói (giây)
    created_at      DATETIME2       DEFAULT GETDATE()
);
GO

-- ============================================================
-- BẢNG 4: meeting_minutes
-- Lưu đường dẫn file biên bản thành phẩm (do Gemini tạo)
-- ============================================================
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='meeting_minutes' AND xtype='U')
CREATE TABLE meeting_minutes (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    meeting_id      INT             NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    minutes_path    NVARCHAR(500)   NOT NULL,           -- Đường dẫn file .md hoặc .txt biên bản
    created_at      DATETIME2       DEFAULT GETDATE()
);
GO

-- ============================================================
-- INDEX để tăng tốc truy vấn
-- ============================================================
CREATE INDEX idx_transcripts_meeting ON transcripts(meeting_id);
CREATE INDEX idx_meeting_code ON meetings(meeting_code);
GO

-- ============================================================
-- VIEW: Xem nhanh toàn bộ transcript một cuộc họp kèm tên người nói
-- ============================================================
CREATE OR ALTER VIEW vw_meeting_transcript AS
SELECT
    m.meeting_code,
    m.title         AS meeting_title,
    m.started_at,
    COALESCE(s.name, t.speaker_label, N'Không rõ') AS speaker_name,
    t.text_content,
    t.text_file_path,
    t.start_time_sec,
    t.end_time_sec,
    t.created_at    AS spoken_at
FROM transcripts t
JOIN meetings m ON t.meeting_id = m.id
LEFT JOIN speakers s ON t.speaker_id = s.id;
GO

PRINT N'Database MeetingMinutesDB đã được khởi tạo thành công!';
GO
