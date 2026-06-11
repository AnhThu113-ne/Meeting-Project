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
          {/* HỌP ONLINE GOOGLE MEET */}
          <button
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
              padding: '0.85rem', borderRadius: '0.75rem', cursor: 'pointer',
              background: 'rgba(255, 255, 255, 0.08)', color: 'white', border: '1px solid rgba(255,255,255,0.08)',
              fontWeight: '700', fontSize: '0.95rem', marginBottom: '0.25rem',
              transition: 'all 0.3s ease'
            }}
            onMouseOver={e => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)'}
            onMouseOut={e => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)'}
            onClick={() => window.open('https://meet.google.com/new', '_blank')}
          >
            <Play size={18} fill="currentColor" />
            Họp Google Meet (Online)
          </button>

          {/* HỌP TRỰC TIẾP VỚI ROBOT AI */}
          <button
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
              padding: '0.85rem', borderRadius: '0.75rem', cursor: 'pointer',
              background: activeTab === 'in-person' ? 'var(--gradient-main)' : 'rgba(255, 255, 255, 0.08)',
              color: 'white', border: activeTab === 'in-person' ? 'none' : '1px solid rgba(255,255,255,0.08)',
              fontWeight: '700', fontSize: '0.95rem', marginBottom: '1rem',
              boxShadow: activeTab === 'in-person' ? '0 6px 12px rgba(124, 58, 237, 0.3)' : 'none',
              transition: 'all 0.3s ease'
            }}
            onMouseOver={e => e.currentTarget.style.transform = 'translateY(-2px)'}
            onMouseOut={e => e.currentTarget.style.transform = 'translateY(0)'}
            onClick={() => setActiveTab('in-person')}
          >
            <Mic size={18} />
            Họp Trực Tiếp (Trợ Lý AI)
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
          {activeTab === 'in-person' && <InPersonMeetingTab key="in-person" />}
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

function InPersonMeetingTab() {
  const [status, setStatus] = useState("standby"); // standby | recording | analyzing | done
  const [transcript, setTranscript] = useState([]);
  const [result, setResult] = useState(null);
  const [tempText, setTempText] = useState("");
  const [isPdfPreview, setIsPdfPreview] = useState(false);
  const [isAiSpeaking, setIsAiSpeaking] = useState(false);
  const [aiReply, setAiReply] = useState("");

  const recognitionRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const isAiSpeakingRef = useRef(false);

  // Trình duyệt load giọng nói bất đồng bộ
  useEffect(() => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.getVoices();
    }
  }, []);

  // Text-To-Speech (TTS) - Tối ưu hóa: gọi Google AI Studio Despina từ backend, fallback sang browser speechSynthesis
  const speak = (text, callbackOnEnd = null) => {
    // 1. Tạm thời tắt nhận dạng giọng nói để mic không thu tiếng loa phát
    setIsAiSpeaking(true);
    isAiSpeakingRef.current = true;
    if (recognitionRef.current) {
      recognitionRef.current.onend = null;
      try {
        recognitionRef.current.stop();
      } catch (e) {}
    }

    // Định nghĩa hàm fallback sang speechSynthesis trình duyệt
    const fallbackToBrowserTTS = () => {
      if (!('speechSynthesis' in window)) {
        finishSpeaking();
        return;
      }
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'vi-VN';

      const voices = window.speechSynthesis.getVoices();
      const viVoices = voices.filter(v => v.lang.toLowerCase().replace('_', '-').includes('vi-vn') || v.lang.toLowerCase().includes('vi'));
      
      let viVoice = viVoices.find(v => v.name.toLowerCase().includes("hoailim") || v.name.toLowerCase().includes("natural"));
      if (!viVoice) {
        viVoice = viVoices.find(v => v.name.toLowerCase().includes("google") || v.name.toLowerCase().includes("online"));
      }
      if (!viVoice && viVoices.length > 0) {
        viVoice = viVoices[0];
      }
      if (viVoice) {
        utterance.voice = viVoice;
      }

      utterance.onend = () => {
        finishSpeaking();
      };
      utterance.onerror = () => {
        finishSpeaking();
      };
      window.speechSynthesis.speak(utterance);
    };

    const finishSpeaking = () => {
      setIsAiSpeaking(false);
      isAiSpeakingRef.current = false;
      if (callbackOnEnd) callbackOnEnd();
      // Bật lại lắng nghe giọng nói
      if (statusRef.current !== "analyzing") {
        initSpeechRecognition();
      }
    };

    // 2. Thử gọi API Backend để lấy giọng Despina từ Google AI Studio
    fetch(`${API_BASE}/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    })
    .then(res => {
      if (!res.ok) throw new Error("API error");
      const contentType = res.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        return res.json().then(data => { throw new Error(data.message || "Lỗi API") });
      }
      return res.blob();
    })
    .then(blob => {
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.onended = () => {
        URL.revokeObjectURL(url);
        finishSpeaking();
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        fallbackToBrowserTTS();
      };
      audio.play().catch(() => {
        fallbackToBrowserTTS();
      });
    })
    .catch(err => {
      console.warn("[TTS] Không thể dùng giọng Despina từ API (chưa cấu hình GEMINI_API_KEY). Fallback sang giọng trình duyệt:", err);
      fallbackToBrowserTTS();
    });
  };

  // Start continuous listening for triggers (STT)
  useEffect(() => {
    initSpeechRecognition();
    return () => {
      stopListening();
    };
  }, []);

  // Sync state reference to use inside async callbacks
  const statusRef = useRef(status);
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  const initSpeechRecognition = () => {
    if (!('webkitSpeechRecognition' in window)) {
      alert("Trình duyệt của bạn không hỗ trợ Speech Recognition. Hãy dùng Google Chrome!");
      return;
    }

    const rec = new webkitSpeechRecognition();
    rec.lang = "vi-VN";
    rec.continuous = true;
    rec.interimResults = true;

    rec.onstart = () => {
      console.log("[AI Assistant] Bắt đầu lắng nghe khẩu lệnh...");
    };

    rec.onerror = (e) => {
      console.warn("[AI Assistant] Lỗi speech:", e.error);
    };

    rec.onend = () => {
      if (statusRef.current !== "analyzing" && !isAiSpeakingRef.current) {
        try { rec.start(); } catch (err) {}
      }
    };

    rec.onresult = (event) => {
      if (isAiSpeakingRef.current) return; // Không xử lý khi AI đang nói
      setAiReply(""); // Xóa phản hồi cũ của AI khi nghe tiếng mới

      let finalSpeech = "";
      let interimSpeech = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const text = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalSpeech += text + " ";
        } else {
          interimSpeech += text;
        }
      }

      const spokenText = finalSpeech.trim();
      if (spokenText) {
        handleSpokenText(spokenText);
      }
      if (interimSpeech) {
        setTempText(interimSpeech);
      }
    };

    recognitionRef.current = rec;
    try { rec.start(); } catch(err) {}
  };

  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.onend = null;
      try { recognitionRef.current.stop(); } catch(e) {}
    }
  };

  const handleSpokenText = (text) => {
    console.log("[AI Assistant] Nghe được:", text);
    const normalizedText = text.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/đ/g, "d");

    // Check Start Trigger: "Bắt đầu cuộc họp"
    if (statusRef.current === "standby") {
      const triggers = ["bat dau cuoc hop", "bat dau thoi", "bat dau nao", "bat dau hop"];
      if (triggers.some(tr => normalizedText.includes(tr))) {
        startMeeting();
        return;
      }

      // Giao tiếp tự nhiên
      if (text.trim()) {
        setTempText(`Bạn nói: "${text}"`);
        fetch(`${API_BASE}/chat-assistant`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        })
        .then(res => res.json())
        .then(data => {
          if (data.status === "ok" && data.reply) {
            setAiReply(data.reply);
            speak(data.reply);
          }
        })
        .catch(err => {
          console.error("Error chat-assistant:", err);
          setAiReply("Xin lỗi, tôi gặp sự cố kết nối.");
          speak("Xin lỗi, tôi gặp sự cố kết nối.");
        });
      }
    }

    // Check Stop Trigger: "Kết thúc cuộc họp"
    if (statusRef.current === "recording") {
      const stopTriggers = ["ket thuc cuoc hop", "ket thuc hop", "dung cuoc hop", "dung ghi"];
      if (stopTriggers.some(tr => normalizedText.includes(tr))) {
        endMeeting();
        return;
      }

      // Save normal spoken line
      setTranscript(prev => [...prev, { speaker: "Bạn", text }]);
      setTempText("");
    }
  };

  const startMeeting = async () => {
    setStatus("recording");
    setTranscript([]);
    setResult(null);
    setAiReply("");
    setTempText("");
    speak("Dạ, tôi đã bắt đầu ghi âm cuộc họp trực tiếp. Chúc mọi người có một buổi họp hiệu quả!");

    // Start Audio Recording using MediaRecorder
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      audioChunksRef.current = [];

      const options = { mimeType: 'audio/webm' };
      const mediaRecorder = new MediaRecorder(stream, options);
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.start(1000); // Record in 1s slices
      mediaRecorderRef.current = mediaRecorder;
      console.log("[AI Assistant] Bắt đầu thu âm audio...");
    } catch (err) {
      console.error("[AI Assistant] Không thể truy cập Micro:", err);
      alert("Không thể ghi âm cuộc họp do chưa cấp quyền micro!");
    }
  };

  const endMeeting = async () => {
    console.log("[AI Assistant] Nhận lệnh kết thúc.");
    setStatus("analyzing");
    speak("Buổi họp đã kết thúc. Tôi đang tiến hành phân tích giọng nói và tạo biên bản, xin vui lòng chờ trong giây lát.");

    // Stop recording
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
    }
    stopListening();

    // Short timeout to let mediaRecorder collect the last chunk
    setTimeout(async () => {
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
      const audioFile = new File([audioBlob], "in_person_meeting.wav", { type: "audio/wav" });
      
      // Upload to backend for speaker identification + minutes generation
      const formData = new FormData();
      formData.append('file', audioFile);

      try {
        const res = await fetch(`${API_BASE}/upload-audio`, {
          method: 'POST',
          body: formData,
        });
        const data = await res.json();
        if (data.status === "ok") {
          setResult(data);
          setStatus("done");
        } else {
          alert("Lỗi khi phân tích cuộc họp: " + (data.error || "Lỗi không xác định"));
          setStatus("standby");
          initSpeechRecognition();
        }
      } catch (err) {
        alert("Lỗi kết nối tới Server AI Offline.");
        setStatus("standby");
        initSpeechRecognition();
      }
    }, 1000);
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }} className="in-person-tab">
      
      {/* PRINT-ONLY AREA */}
      {result && (
        <div className="print-only-container">
          <div className="pdf-header">
            <h1>BIÊN BẢN CUỘC HỌP CHÍNH THỨC</h1>
            <p>ID Cuộc họp: {result.meeting_id || result.file_id}</p>
            <p>Ngày tạo: {new Date().toLocaleString('vi-VN')}</p>
          </div>
          <hr />
          <div className="pdf-section">
            <h2>Nội Dung Biên Bản</h2>
            <div style={{ whiteSpace: 'pre-wrap' }}>{result.minutes}</div>
          </div>
          <div className="pdf-section" style={{ pageBreakBefore: 'always' }}>
            <h2>Bản Ghi Chi Tiết Cuộc Họp</h2>
            {result.transcript.map((t, idx) => (
              <p key={idx}><strong>{t.speaker}:</strong> {t.text}</p>
            ))}
          </div>
        </div>
      )}

      {/* SCREEN AREA */}
      <div className="screen-only-container" style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        <header style={{ marginBottom: '1.5rem' }}>
          <h1 style={{ fontSize: '2.5rem' }}>Họp Trực Tiếp <span className="gradient-text">Với Trợ Lý AI</span></h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Nói <strong style={{ color: 'var(--accent-primary)' }}>"Bắt đầu cuộc họp"</strong> để ghi âm và bóc băng tự động. Nói <strong style={{ color: '#ef4444' }}>"Kết thúc cuộc họp"</strong> để tạo biên bản.
          </p>
        </header>

        <div style={{ display: 'grid', gridTemplateColumns: status === "done" ? "1fr 400px" : "1fr", gap: '1.5rem', flex: 1, minHeight: 0 }}>
          
          {/* Main Visualizer Area */}
          {status !== "done" ? (
            <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '3rem', position: 'relative' }}>
              
              {/* Robot Assistant Glowing Orb */}
              <div className={`ai-orb ${status === 'recording' ? 'ai-orb-recording' : status === 'analyzing' ? 'ai-orb-loading' : ''}`}>
                <div className="ai-orb-inner">
                  <Mic size={48} color="white" />
                </div>
              </div>

              <h2 style={{ marginTop: '2.5rem', textTransform: 'uppercase', letterSpacing: '2px', fontSize: '1.5rem' }}>
                {status === "standby" && "Đang chờ lệnh khởi động..."}
                {status === "recording" && "Đang ghi nhận cuộc họp..."}
                {status === "analyzing" && "AI Đang xử lý biên bản..."}
              </h2>
              
              <p style={{ color: 'var(--text-secondary)', marginTop: '0.75rem', fontSize: '1rem', textAlign: 'center', maxWidth: '500px' }}>
                {status === "standby" && 'Vui lòng nói rõ "Bắt đầu cuộc họp" hoặc bấm nút thủ công bên dưới.'}
                {status === "recording" && 'Nói "Kết thúc cuộc họp" để kết thúc cuộc họp trực tiếp.'}
                {status === "analyzing" && "Hệ thống đang chạy sinh trắc học nhận dạng giọng nói từng người và tóm tắt cuộc họp..."}
              </p>

              {/* Real-time Subtitle Overlay */}
              {(status === "recording" || status === "standby") && (
                <div className="interim-text-container" style={{ minHeight: '60px', width: '80%', display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: '1.5rem' }}>
                  <p className="interim-text" style={{ color: status === 'recording' ? '#10b981' : '#38bdf8' }}>
                    {status === "recording" ? (tempText || "Đang nghe...") : (aiReply || tempText || "Hãy nói điều gì đó để trò chuyện...")}
                  </p>
                </div>
              )}

              {/* Control Buttons */}
              <div style={{ display: 'flex', gap: '1rem', marginTop: '2.5rem' }}>
                {status === "standby" && (
                  <button className="btn-primary" style={{ padding: '1rem 2.5rem' }} onClick={startMeeting}>
                    Bắt đầu ghi họp
                  </button>
                )}
                {status === "recording" && (
                  <button className="btn-primary" style={{ background: '#ef4444', padding: '1rem 2.5rem' }} onClick={endMeeting}>
                    Kết thúc cuộc họp
                  </button>
                )}
              </div>
            </div>
          ) : (
            
            // Result Display Tab
            <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--glass-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3>Kết quả cuộc họp trực tiếp</h3>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button className="btn-primary" style={{ background: 'rgba(255,255,255,0.1)', padding: '0.5rem 1rem', fontSize: '0.85rem' }} onClick={() => setIsPdfPreview(true)}>
                    Xem trước PDF
                  </button>
                  <button className="btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }} onClick={handlePrint}>
                    Xuất file PDF
                  </button>
                  <button className="btn-primary" style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#ef4444', padding: '0.5rem 1rem', fontSize: '0.85rem' }} onClick={() => { setStatus("standby"); setResult(null); setTranscript([]); initSpeechRecognition(); }}>
                    Họp mới
                  </button>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', flex: 1, minHeight: 0, padding: '1.5rem', overflow: 'hidden' }}>
                
                {/* Transcript Panel */}
                <div style={{ display: 'flex', flexDirection: 'column', background: 'rgba(0,0,0,0.2)', borderRadius: '0.75rem', border: '1px solid var(--glass-border)', overflow: 'hidden' }}>
                  <div style={{ padding: '0.75rem', borderBottom: '1px solid var(--glass-border)', fontWeight: '600' }}>
                    Transcript (Nhận diện & Phân tách giọng nói)
                  </div>
                  <div style={{ flex: 1, overflowY: 'auto', padding: '1rem' }}>
                    {result.transcript.map((t, idx) => (
                      <div key={idx} className="transcript-bubble" style={{ background: 'rgba(255,255,255,0.03)' }}>
                        <div className="speaker-tag speaker-nam">{t.speaker}</div>
                        <p>{t.text}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Minutes Panel */}
                <div style={{ display: 'flex', flexDirection: 'column', background: 'rgba(0,0,0,0.2)', borderRadius: '0.75rem', border: '1px solid var(--glass-border)', overflow: 'hidden' }}>
                  <div style={{ padding: '0.75rem', borderBottom: '1px solid var(--glass-border)', fontWeight: '600' }}>
                    Biên bản cuộc họp (AI tạo lập)
                  </div>
                  <div style={{ flex: 1, overflowY: 'auto', padding: '1.25rem', whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
                    {result.minutes}
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* Real-time speech log on the right side if in recording state */}
          {status === "recording" && (
            <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--glass-border)' }}>
                <h3>Nội dung đang ghi nhận...</h3>
              </div>
              <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {transcript.map((t, idx) => (
                  <div key={idx} className="transcript-bubble animate-fade-in" style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '0.75rem' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--accent-primary)', fontWeight: '700', marginBottom: '0.2rem' }}>{t.speaker}</div>
                    <p style={{ margin: 0, fontSize: '0.9rem' }}>{t.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>

      {/* PDF PREVIEW MODAL */}
      {isPdfPreview && result && (
        <div className="modal-overlay" onClick={() => setIsPdfPreview(false)}>
          <div className="modal-content glass-panel" onClick={e => e.stopPropagation()} style={{ maxWidth: '800px', width: '90%', height: '80%', display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' }}>
            <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--glass-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3>Xem trước bản in PDF</h3>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }} onClick={handlePrint}>
                  Tải xuống / In PDF
                </button>
                <button className="btn-primary" style={{ background: 'rgba(255,255,255,0.1)', padding: '0.5rem 1rem', fontSize: '0.85rem' }} onClick={() => setIsPdfPreview(false)}>
                  Đóng
                </button>
              </div>
            </div>
            
            <div style={{ flex: 1, overflowY: 'auto', padding: '2.5rem', background: '#ffffff', color: '#1e293b', fontFamily: 'Inter, sans-serif' }}>
              <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                <h1 style={{ fontSize: '1.8rem', fontWeight: '800', color: '#0f172a', margin: '0 0 0.5rem 0' }}>BIÊN BẢN CUỘC HỌP</h1>
                <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>Mã cuộc họp: {result.meeting_id || result.file_id}</p>
                <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>Ngày tạo: {new Date().toLocaleString('vi-VN')}</p>
              </div>
              <hr style={{ border: 0, borderTop: '2px solid #e2e8f0', marginBottom: '2rem' }} />
              
              <div style={{ marginBottom: '2rem' }}>
                <h3 style={{ borderBottom: '1px solid #cbd5e1', paddingBottom: '0.5rem', color: '#0f172a' }}>I. BIÊN BẢN AI TÓM TẮT</h3>
                <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.7', fontSize: '0.95rem' }}>{result.minutes}</div>
              </div>
              
              <div>
                <h3 style={{ borderBottom: '1px solid #cbd5e1', paddingBottom: '0.5rem', color: '#0f172a' }}>II. BẢN GHI HỘI THOẠI CHI TIẾT (DIARIZATION)</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
                  {result.transcript.map((t, idx) => (
                    <div key={idx} style={{ padding: '0.5rem 0', borderBottom: '1px dotted #f1f5f9' }}>
                      <span style={{ fontWeight: 'bold', color: '#4338ca', marginRight: '0.5rem' }}>{t.speaker}:</span>
                      <span style={{ fontSize: '0.95rem' }}>{t.text}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

    </motion.div>
  );
}

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
      if (Array.isArray(data)) {
        setSpeakers(data);
      } else {
        setSpeakers([]);
        console.error("fetchSpeakers expected array, got:", data);
      }
    } catch (err) {
      console.error(err);
      setSpeakers([]);
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
  const [isPdfPreview, setIsPdfPreview] = useState(false);

  useEffect(() => {
    fetchMeetings();
  }, []);

  const fetchMeetings = async () => {
    try {
      const res = await fetch(`${API_BASE}/meetings`);
      const data = await res.json();
      if (Array.isArray(data)) {
        setMeetings(data);
      } else {
        setMeetings([]);
        console.error("fetchMeetings expected array, got:", data);
      }
    } catch (err) {
      console.error(err);
      setMeetings([]);
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

  const handleDeleteMeeting = async (id) => {
    if (!window.confirm("Bạn có chắc chắn muốn xóa hoàn toàn cuộc họp này? Thao tác này sẽ xóa tất cả file âm thanh và biên bản liên quan.")) {
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/meeting/${id}`, {
        method: 'DELETE'
      });
      const data = await res.json();
      if (data.status === "ok") {
        alert("Đã xóa cuộc họp thành công!");
        setSelectedMeeting(null);
        fetchMeetings(); // Tải lại danh sách cuộc họp
      } else {
        alert("Lỗi khi xóa cuộc họp: " + data.message);
      }
    } catch (err) {
      console.error(err);
      alert("Lỗi kết nối khi xóa cuộc họp");
    }
  };

  if (selectedMeeting && meetingDetails) {
    return (
      <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        
        {/* PRINT-ONLY AREA */}
        <div className="print-only-container">
          <div className="pdf-header">
            <h1>BIÊN BẢN CUỘC HỌP CHÍNH THỨC</h1>
            <p>ID Cuộc họp: {selectedMeeting}</p>
            <p>Ngày tạo: {meetingDetails.started_at && meetingDetails.started_at !== 'None' ? new Date(meetingDetails.started_at).toLocaleString('vi-VN') : new Date().toLocaleString('vi-VN')}</p>
          </div>
          <hr />
          <div className="pdf-section">
            <h2>Nội Dung Biên Bản</h2>
            <div style={{ whiteSpace: 'pre-wrap' }}>{meetingDetails.minutes}</div>
          </div>
          <div className="pdf-section" style={{ pageBreakBefore: 'always' }}>
            <h2>Bản Ghi Chi Tiết Cuộc Họp</h2>
            {meetingDetails.transcript.map((t, idx) => (
              <p key={idx}><strong>{t.speaker}:</strong> {t.text}</p>
            ))}
          </div>
        </div>

        {/* SCREEN AREA */}
        <div className="screen-only-container" style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
          <header style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <button
                className="btn-primary"
                style={{ padding: '0.5rem', borderRadius: '50%', width: 40, height: 40, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                onClick={() => { setSelectedMeeting(null); setIsPdfPreview(false); }}
              >
                ←
              </button>
              <div>
                <h1 style={{ fontSize: '2rem' }}>Chi tiết <span className="gradient-text">Biên Bản</span></h1>
                <p style={{ color: 'var(--text-secondary)' }}>ID Cuộc họp: {selectedMeeting}</p>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button className="btn-primary" style={{ background: 'rgba(255,255,255,0.1)', padding: '0.6rem 1.5rem', fontSize: '0.9rem' }} onClick={() => setIsPdfPreview(true)}>
                Xem trước PDF
              </button>
              <button className="btn-primary" style={{ padding: '0.6rem 1.5rem', fontSize: '0.9rem' }} onClick={() => window.print()}>
                Xuất file PDF
              </button>
              <button
                className="btn-primary"
                style={{ background: '#ef4444', padding: '0.6rem 1.5rem', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                onClick={() => handleDeleteMeeting(selectedMeeting)}
              >
                Xóa cuộc họp
              </button>
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
        </div>

        {/* PDF PREVIEW MODAL */}
        {isPdfPreview && (
          <div className="modal-overlay" onClick={() => setIsPdfPreview(false)}>
            <div className="modal-content glass-panel" onClick={e => e.stopPropagation()} style={{ maxWidth: '800px', width: '90%', height: '80%', display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' }}>
              <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--glass-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3>Xem trước bản in PDF</h3>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button className="btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }} onClick={() => window.print()}>
                    Tải xuống / In PDF
                  </button>
                  <button className="btn-primary" style={{ background: 'rgba(255,255,255,0.1)', padding: '0.5rem 1rem', fontSize: '0.85rem' }} onClick={() => setIsPdfPreview(false)}>
                    Đóng
                  </button>
                </div>
              </div>
              
              <div style={{ flex: 1, overflowY: 'auto', padding: '2.5rem', background: '#ffffff', color: '#1e293b', fontFamily: 'Inter, sans-serif' }}>
                <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                  <h1 style={{ fontSize: '1.8rem', fontWeight: '800', color: '#0f172a', margin: '0 0 0.5rem 0' }}>BIÊN BẢN CUỘC HỌP</h1>
                  <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>Mã cuộc họp: {selectedMeeting}</p>
                  <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>Ngày tạo: {meetingDetails.started_at && meetingDetails.started_at !== 'None' ? new Date(meetingDetails.started_at).toLocaleString('vi-VN') : new Date().toLocaleString('vi-VN')}</p>
                </div>
                <hr style={{ border: 0, borderTop: '2px solid #e2e8f0', marginBottom: '2rem' }} />
                
                <div style={{ marginBottom: '2rem' }}>
                  <h3 style={{ borderBottom: '1px solid #cbd5e1', paddingBottom: '0.5rem', color: '#0f172a' }}>I. BIÊN BẢN AI TÓM TẮT</h3>
                  <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.7', fontSize: '0.95rem', color: '#1e293b' }}>{meetingDetails.minutes}</div>
                </div>
                
                <div>
                  <h3 style={{ borderBottom: '1px solid #cbd5e1', paddingBottom: '0.5rem', color: '#0f172a' }}>II. BẢN GHI HỘI THOẠI CHI TIẾT (DIARIZATION)</h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
                    {meetingDetails.transcript.map((t, idx) => (
                      <div key={idx} style={{ padding: '0.5rem 0', borderBottom: '1px dotted #f1f5f9' }}>
                        <span style={{ fontWeight: 'bold', color: '#4338ca', marginRight: '0.5rem' }}>{t.speaker}:</span>
                        <span style={{ fontSize: '0.95rem', color: '#1e293b' }}>{t.text}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
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
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>ID: {m.id} • Mã phòng: {m.meeting_code} • Bắt đầu: {m.started_at && m.started_at !== 'None' ? new Date(m.started_at).toLocaleString('vi-VN') : 'Chưa bắt đầu'} • Trạng thái: {m.status}</p>
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
