"""
APIRouter: /meetings endpoints
"""
import os
import asyncio
import aiofiles
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime

from backend.db.database import get_db, Meeting, Transcript, MeetingMinutes, SessionLocal

router = APIRouter(prefix="/api/meetings", tags=["meetings"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".mp4", ".webm"}


@router.post("/upload")
async def upload_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type {ext} not supported. Use: {', '.join(ALLOWED_EXTENSIONS)}")

    # Unique filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    # Save file
    async with aiofiles.open(file_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            await out.write(chunk)

    title = os.path.splitext(file.filename)[0]
    meeting = Meeting(
        title=title,
        filename=file.filename,
        file_path=file_path,
        status="pending"
    )
    db.add(meeting)
    db.commit()
    db.refresh(meeting)

    # Kick off background AI processing
    background_tasks.add_task(run_ai_pipeline, meeting.id, file_path)

    return {"meeting_id": meeting.id, "title": title, "status": "pending"}


def run_ai_pipeline(meeting_id: int, audio_path: str):
    """Synchronous wrapper to run async processor."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        from backend.core.processor import process_meeting
        loop.run_until_complete(process_meeting(meeting_id, audio_path, SessionLocal))
    finally:
        loop.close()


@router.get("")
def list_meetings(db: Session = Depends(get_db)):
    meetings = db.query(Meeting).order_by(Meeting.created_at.desc()).all()
    return [
        {
            "id": m.id,
            "title": m.title,
            "filename": m.filename,
            "duration": round(m.duration or 0, 1),
            "status": m.status,
            "created_at": m.created_at.isoformat() if m.created_at else None
        }
        for m in meetings
    ]


@router.get("/{meeting_id}")
def get_meeting(meeting_id: int, db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(404, "Meeting not found")

    transcripts = db.query(Transcript).filter(
        Transcript.meeting_id == meeting_id
    ).order_by(Transcript.start_time).all()

    minutes = db.query(MeetingMinutes).filter(
        MeetingMinutes.meeting_id == meeting_id
    ).first()

    return {
        "id": meeting.id,
        "title": meeting.title,
        "filename": meeting.filename,
        "duration": round(meeting.duration or 0, 1),
        "status": meeting.status,
        "created_at": meeting.created_at.isoformat() if meeting.created_at else None,
        "transcripts": [
            {
                "speaker": t.speaker,
                "text": t.text,
                "start": round(t.start_time, 1),
                "end": round(t.end_time, 1)
            }
            for t in transcripts
        ],
        "summary": minutes.summary if minutes else None
    }


@router.delete("/{meeting_id}")
def delete_meeting(meeting_id: int, db: Session = Depends(get_db)):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    if os.path.exists(meeting.file_path):
        os.remove(meeting.file_path)
    db.delete(meeting)
    db.commit()
    return {"message": "Deleted successfully"}


@router.patch("/{meeting_id}/rename-speaker")
def rename_speaker(
    meeting_id: int,
    body: dict,
    db: Session = Depends(get_db)
):
    """
    Doi ten mot speaker trong cuoc hop.
    Body: { "old_name": "SPEAKER_00", "new_name": "Thu" }
    """
    old_name = body.get("old_name", "").strip()
    new_name = body.get("new_name", "").strip()

    if not old_name or not new_name:
        raise HTTPException(400, "old_name va new_name la bat buoc")

    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(404, "Meeting not found")

    # Cap nhat ten trong tat ca transcript segments
    updated = db.query(Transcript).filter(
        Transcript.meeting_id == meeting_id,
        Transcript.speaker == old_name
    ).all()

    count = len(updated)
    for t in updated:
        t.speaker = new_name
    db.commit()

    return {
        "message": f"Da doi '{old_name}' → '{new_name}' ({count} dong transcript)",
        "updated_count": count
    }


@router.get("/{meeting_id}/speakers")
def get_meeting_speakers(meeting_id: int, db: Session = Depends(get_db)):
    """Lay danh sach cac speaker duy nhat trong cuoc hop."""
    transcripts = db.query(Transcript.speaker).filter(
        Transcript.meeting_id == meeting_id
    ).distinct().all()
    speakers = sorted(set(t.speaker for t in transcripts if t.speaker))
    return {"speakers": speakers}
