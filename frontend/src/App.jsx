import React, { useState, useEffect, useRef } from 'react';
import {
  Mic, FileAudio, LayoutDashboard, UserPlus,
  Settings, History, Upload, CheckCircle2,
  Loader2, Play, Users, MessageSquare, ClipboardList,
  StopCircle, ArrowRight
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = "http://localhost:8000";

function App() {
  const [activeTab, setActiveTab] = useState('voice');

  return (
    <div className="app-container">
      {/* SIDEBAR */}
      <aside className="sidebar glass-panel">
        <div className="logo-section">
          <h2 className="gradient-text" style={{ fontSize: '1.5rem', marginBottom: '2rem' }}>AI Meeting</h2>
        </div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {/* NÚT TẠO CUỘC HỌP MỚI VÀO THẲNG GOOGLE MEET */}
          <button
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
              padding: '1rem', borderRadius: '0.75rem', cursor: 'pointer',
              background: 'var(--gradient-main)', color: 'white', border: 'none',
              fontWeight: '700', fontSize: '1rem', marginBottom: '1rem',
              boxShadow: '0 8px 16px rgba(124, 58, 237, 0.4)', transition: 'all 0.3s ease'
            }}
            onMouseOver={e => e.currentTarget.style.transform = 'translateY(-2px)'}
            onMouseOut={e => e.currentTarget.style.transform = 'translateY(0)'}
            onClick={() => window.open('https://meet.google.com/new', '_blank')}
          >
            <Play size={20} fill="currentColor" />
            Tạo phòng Meet ngay
          </button>

          <SidebarItem icon={<LayoutDashboard size={20} />} label="Dashboard (Upload)" active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} />
          <SidebarItem icon={<UserPlus size={20} />} label="Đăng ký giọng nói" active={activeTab === 'voice'} onClick={() => setActiveTab('voice')} />
          <SidebarItem icon={<History size={20} />} label="Lịch sử cuộc họp" active={activeTab === 'history'} onClick={() => setActiveTab('history')} />
        </nav>

        <div style={{ marginTop: 'auto', padding: '1rem', background: 'rgba(255,255,255,0.05)', borderRadius: '1rem' }}>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Trạng thái hệ thống</p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981' }}></div>
            <span style={{ fontSize: '0.85rem' }}>Online (FastAPI)</span>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-content">
        <AnimatePresence mode="wait">
          {activeTab === 'dashboard' && <DashboardTab key="dashboard" />}
          {activeTab === 'voice' && <VoiceRegistrationTab key="voice" />}
          {activeTab === 'history' && <HistoryTab key="history" />}
        </AnimatePresence>
      </main>
    </div>
  );
}

// ==========================================
// TABS
// ==========================================

function DashboardTab() {
  const [uploading, setUploading] = useState(false);
  const [processingId, setProcessingId] = useState(null);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");

  useEffect(() => {
    let interval;
    if (processingId && status === "processing") {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/result/${processingId}`);
          const data = await res.json();
          if (data.status !== "not_ready" && data.status !== "error") {
            setResult(data);
            setStatus("done");
            setProcessingId(null);
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [processingId, status]);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setStatus("uploading");

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/upload-audio`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (data.status === "ok") {
        setResult({
          transcript: data.transcript || [],
          minutes: data.minutes || "Không có nội dung."
        });
        setStatus("done");
      } else {
        alert("Lỗi khi xử lý âm thanh: " + (data.error || "Unknown error"));
        setStatus("idle");
      }
    } catch (err) {
      alert("Lỗi kết nối tới Server AI Offline.");
      setStatus("idle");
    } finally {
      setUploading(false);
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <header style={{ marginBottom: '1.5rem' }}>
        <h1 style={{ fontSize: '2.5rem' }}>Phân tích <span className="gradient-text">Âm Thanh</span></h1>
        <p style={{ color: 'var(--text-secondary)' }}>Tải lên file ghi âm (.mp3, .wav) để AI nhận diện người nói và tóm tắt theo giọng từng người.</p>
      </header>

      {status === "idle" && (
        <div
          className="glass-panel"
          style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', border: '2px dashed var(--glass-border)', cursor: 'pointer' }}
          onClick={() => document.getElementById('audio-upload').click()}
        >
          <div style={{ padding: '2rem', borderRadius: '50%', background: 'rgba(124, 58, 237, 0.1)', color: 'var(--accent-primary)', marginBottom: '1.5rem' }}>
            <Upload size={48} />
          </div>
          <h3>Tải lên tệp âm thanh</h3>
          <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Hệ thống sẽ dùng Viettel STT và Pyannote để phân tách giọng nói</p>
          <input type="file" id="audio-upload" hidden onChange={handleFileUpload} accept="audio/*" />
        </div>
      )}

      {status === "uploading" || status === "processing" ? (
        <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <Loader2 size={48} className="animate-spin" style={{ color: 'var(--accent-primary)', marginBottom: '1rem' }} />
          <h3>AI đang xử lý âm thanh...</h3>
          <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Có thể mất vài phút tùy vào độ dài file ghi âm.</p>
        </div>
      ) : null}

      {status === "done" && result && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 400px', gap: '1.5rem', flex: 1 }}>
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--glass-border)' }}>
              <h3>Transcript (bóc băng)</h3>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem' }}>
              {result.transcript.map((turn, i) => (
                <div key={i} className={`transcript-bubble`} style={{ background: 'rgba(255,255,255,0.03)' }}>
                  <div className="speaker-tag speaker-nam">{turn.speaker}</div>
                  <p>{turn.text}</p>
                </div>
              ))}
            </div>
          </div>
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--glass-border)' }}>
              <h3>Biên bản cuộc họp</h3>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', whiteSpace: 'pre-wrap' }}>
              {result.minutes}
            </div>
            <div style={{ padding: '1rem' }}>
              <button className="btn-primary" style={{ width: '100%' }} onClick={() => setStatus("idle")}>Phân tích file khác</button>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}

function VoiceRegistrationTab() {
  const [name, setName] = useState("");
  const [file, setFile] = useState(null);
  const [recording, setRecording] = useState(false);
  const [registering, setRegistering] = useState(false);
  const [success, setSuccess] = useState("");
  const [speakers, setSpeakers] = useState([]);
  const [previewUrl, setPreviewUrl] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  useEffect(() => {
    fetchSpeakers();
  }, []);

  const fetchSpeakers = async () => {
    try {
      const res = await fetch(`${API_BASE}/speakers`);
      const data = await res.json();
      setSpeakers(data);
    } catch (err) {
      console.error(err);
    }
  };

  const startRecord = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const audioFile = new File([audioBlob], "recorded_voice.wav", { type: "audio/wav" });
        setFile(audioFile);
        setPreviewUrl(URL.createObjectURL(audioBlob));
      };

      mediaRecorderRef.current.start();
      setRecording(true);
    } catch (err) {
      alert("Không thể truy cập Micro!");
    }
  };

  const stopRecord = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  };

  const handleRegister = async () => {
    if (!name.trim()) return alert("Vui lòng nhập tên diễn giả!");
    if (!file) return alert("Vui lòng ghi âm hoặc chọn file giọng nói mẫu!");

    setRegistering(true);
    setSuccess("");
    const fd = new FormData();
    fd.append("name", name);
    fd.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/register-speaker`, {
        method: 'POST',
        body: fd
      });
      const data = await res.json();
      if (data.status === "ok") {
        setSuccess("Đã đăng ký thành công giọng nói cho " + name);
        setName("");
        setFile(null);
        setPreviewUrl(null);
        fetchSpeakers();
      }
    } catch (err) {
      alert("Lỗi khi kết nối đến API");
    } finally {
      setRegistering(false);
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <header style={{ marginBottom: '1.5rem' }}>
        <h1 style={{ fontSize: '2.5rem' }}>Đăng ký <span className="gradient-text">Giọng Nói</span></h1>
        <p style={{ color: 'var(--text-secondary)' }}>AI cần một đoạn âm thanh mẫu (10-15s) để có thể nhận diện bạn trong các cuộc họp.</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', flex: 1 }}>
        {/* Form Đăng ký */}
        <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>Tên hiển thị trong biên bản</label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="VD: Nguyễn Văn A, Giám đốc..."
              style={{
                width: '100%', padding: '1rem', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--glass-border)',
                borderRadius: '0.75rem', color: 'white', outline: 'none', fontSize: '1rem'
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500' }}>Giọng nói mẫu</label>
            <div style={{ display: 'flex', gap: '1rem' }}>
              <button
                className={`btn-primary ${recording ? 'recording-anim' : ''}`}
                style={{ flex: 1, background: recording ? '#ef4444' : 'rgba(255,255,255,0.1)', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
                onClick={recording ? stopRecord : startRecord}
              >
                {recording ? <StopCircle /> : <Mic />}
                {recording ? "Dừng ghi âm" : "Micro: Ghi âm trực tiếp"}
              </button>

              <button
                style={{ flex: 1, background: 'rgba(255,255,255,0.1)', color: 'white', border: 'none', borderRadius: '0.75rem', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', fontWeight: '600' }}
                onClick={() => document.getElementById('voice-upload').click()}
              >
                <Upload size={18} />
                Hoặc tải lên file Audio
              </button>
              <input type="file" id="voice-upload" hidden accept="audio/*" onChange={e => {
                const f = e.target.files[0];
                if (f) {
                  setFile(f);
                  setPreviewUrl(URL.createObjectURL(f));
                }
              }} />
            </div>

            {file && (
              <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '0.75rem', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                <p style={{ color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', wordBreak: 'break-all' }}>
                  <CheckCircle2 size={16} /> Đã có file: {file.name || 'Ghi âm trực tiếp'}
                </p>
                {previewUrl && (
                  <audio controls src={previewUrl} style={{ width: '100%', height: '36px', outline: 'none', borderRadius: '4px' }} />
                )}
              </div>
            )}
          </div>

          <div style={{ marginTop: 'auto' }}>
            <button
              className="btn-primary"
              style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5rem', fontSize: '1.1rem', padding: '1rem' }}
              onClick={handleRegister}
              disabled={registering}
            >
              {registering ? <Loader2 className="animate-spin" /> : <UserPlus />}
              {registering ? "Đang xử lý đăng ký..." : "Đăng ký giọng nói cho AI"}
            </button>
            {success && <p style={{ color: '#10b981', marginTop: '1rem', textAlign: 'center' }}>{success}</p>}
          </div>

        </div>

        {/* Danh sách đã đăng ký */}
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--glass-border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Users size={20} className="speaker-nam" /> Danh sách đã đăng ký</h3>
            <span style={{ background: 'rgba(255,255,255,0.1)', padding: '0.2rem 0.5rem', borderRadius: '1rem', fontSize: '0.8rem' }}>{speakers.length} người</span>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {speakers.length === 0 ? (
              <p style={{ textAlign: 'center', color: 'var(--text-secondary)', marginTop: '2rem' }}>Chưa có diễn giả nào đăng ký</p>
            ) : (
              speakers.map((s, idx) => (
                <div key={idx} style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <h4 style={{ fontSize: '1.1rem', marginBottom: '0.2rem' }}>{s.name}</h4>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>ID: {s.id} • Ngày tạo: {new Date(s.created_at).toLocaleDateString()}</p>
                  </div>
                  <div style={{ width: 40, height: 40, borderRadius: '50%', background: 'rgba(124, 58, 237, 0.2)', color: 'var(--accent-primary)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Mic size={18} />
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function HistoryTab() {
  const [meetings, setMeetings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedMeeting, setSelectedMeeting] = useState(null);
  const [meetingDetails, setMeetingDetails] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);

  useEffect(() => {
    fetchMeetings();
  }, []);

  const fetchMeetings = async () => {
    try {
      const res = await fetch(`${API_BASE}/meetings`);
      const data = await res.json();
      setMeetings(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadMeetingDetails = async (id) => {
    setSelectedMeeting(id);
    setLoadingDetails(true);
    try {
      const res = await fetch(`${API_BASE}/meeting/${id}/details`);
      const data = await res.json();
      setMeetingDetails(data);
    } catch (err) {
      console.error(err);
      alert("Lỗi khi tải chi tiết cuộc họp");
      setSelectedMeeting(null);
    } finally {
      setLoadingDetails(false);
    }
  };

  if (selectedMeeting && meetingDetails) {
    return (
      <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <header style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button
            className="btn-primary"
            style={{ padding: '0.5rem', borderRadius: '50%', width: 40, height: 40, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            onClick={() => setSelectedMeeting(null)}
          >
            ←
          </button>
          <div>
            <h1 style={{ fontSize: '2rem' }}>Chi tiết <span className="gradient-text">Biên Bản</span></h1>
            <p style={{ color: 'var(--text-secondary)' }}>ID Cuộc họp: {selectedMeeting}</p>
          </div>
        </header>

        {meetingDetails.audio_path && (
          <div style={{ marginBottom: '1.5rem', background: 'rgba(255,255,255,0.05)', padding: '1rem', borderRadius: '1rem', border: '1px solid var(--glass-border)' }}>
            <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Mic size={16} className="speaker-nam" /> File Ghi Âm Gốc Toàn Cuộc Họp
            </h3>
            <audio
              controls
              src={`${API_BASE}/static/voice/${meetingDetails.audio_path.split(/[\\/]/).pop()}`}
              style={{ width: '100%', outline: 'none', height: 40 }}
            />
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 400px', gap: '1.5rem', flex: 1, minHeight: 0 }}>
          {/* Transcript Panel */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--glass-border)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <MessageSquare size={20} className="speaker-nam" />
              <h3 style={{ fontSize: '1.1rem' }}>Bản ghi hội thoại chi tiết</h3>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem' }}>
              {meetingDetails.transcript.length === 0 ? (
                <p style={{ color: 'var(--text-secondary)' }}>Không có dữ liệu hội thoại.</p>
              ) : (
                meetingDetails.transcript.map((turn, i) => (
                  <div key={i} className={`transcript-bubble animate-fade-in`} style={{ background: turn.speaker !== 'Unknown' ? 'rgba(14, 165, 233, 0.1)' : 'rgba(255,255,255,0.03)' }}>
                    <div className="speaker-tag speaker-nam">{turn.speaker}</div>
                    <p>{turn.text}</p>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* AI Minutes Panel */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--glass-border)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <ClipboardList size={20} style={{ color: '#facc15' }} />
              <h3 style={{ fontSize: '1.1rem' }}>Biên bản cuộc họp</h3>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', lineHeight: '1.8' }}>
              <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit', color: '#e2e8f0' }}>{meetingDetails.minutes}</pre>
            </div>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <header style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 style={{ fontSize: '2.5rem' }}>Lịch sử <span className="gradient-text">Cuộc họp</span></h1>
          <p style={{ color: 'var(--text-secondary)' }}>Xem lại biên bản và bản ghi của các cuộc họp đã qua.</p>
        </div>
        <button className="btn-primary" onClick={fetchMeetings} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem' }}>
          <History size={18} /> Làm mới
        </button>
      </header>

      <div className="glass-panel" style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
            <Loader2 size={32} className="animate-spin" style={{ color: 'var(--accent-primary)' }} />
          </div>
        ) : meetings.length === 0 ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
            <History size={48} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
            <h3>Chưa có cuộc họp nào</h3>
            <p>Hãy bắt đầu ghi cuộc họp từ Extension để dữ liệu xuất hiện ở đây.</p>
          </div>
        ) : (
          meetings.map(m => (
            <div key={m.id} style={{
              background: 'rgba(255,255,255,0.03)', border: '1px solid var(--glass-border)',
              padding: '1.25rem', borderRadius: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center'
            }}>
              <div>
                <h3 style={{ fontSize: '1.2rem', marginBottom: '0.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {m.title || `Cuộc họp ${m.meeting_code}`}
                  {m.has_minutes && <span style={{ background: '#10b98122', color: '#10b981', fontSize: '0.7rem', padding: '0.2rem 0.5rem', borderRadius: '1rem' }}>Có biên bản</span>}
                </h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>ID: {m.id} • Mã phòng: {m.meeting_code} • Bắt đầu: {new Date(m.started_at).toLocaleString('vi-VN')} • Trạng thái: {m.status}</p>
              </div>
              <button
                className="btn-primary"
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1.25rem' }}
                onClick={() => loadMeetingDetails(m.id)}
                disabled={loadingDetails && selectedMeeting === m.id}
              >
                {loadingDetails && selectedMeeting === m.id ? <Loader2 size={16} className="animate-spin" /> : "Xem chi tiết"}
                <ArrowRight size={16} />
              </button>
            </div>
          ))
        )}
      </div>
    </motion.div>
  );
}

function SidebarItem({ icon, label, active, onClick }) {
  return (
    <div
      onClick={onClick}
      style={{
        display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.85rem 1rem', borderRadius: '0.75rem', cursor: 'pointer',
        background: active ? 'var(--gradient-main)' : 'transparent',
        boxShadow: active ? '0 4px 12px rgba(124, 58, 237, 0.3)' : 'none',
        transition: 'all 0.2s ease',
        color: active ? 'white' : 'var(--text-secondary)'
      }}
    >
      {icon}
      <span style={{ fontWeight: active ? '600' : '500', fontSize: '0.95rem' }}>{label}</span>
    </div>
  );
}

export default App;
