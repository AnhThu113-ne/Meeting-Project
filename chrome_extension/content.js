// content.js - Google Meet AI Meeting Minutes
// Su dung Web Speech API (tich hop san trong Chrome, ho tro tieng Viet)
// Khong can Viettel STT, khong can gui audio file

const BACKEND = "http://localhost:8000";

let recognition    = null;
let roomId         = null;
let state          = "standby"; // standby | recording | finalizing
let badgeEl        = null;
let transcript     = [];
let dbMeetingId    = null;
let isListening    = false;

// Lay ten dien gia hien tai (Luu vao localStorage de do phai nhap lai nhieu lan)
let mySpeakerName = localStorage.getItem("ai_meet_speaker_name");
if (!mySpeakerName) {
  mySpeakerName = prompt("Vui lòng nhập tên của bạn (Tên này sẽ hiển thị trong Biên bản AI):", "Nam");
  if (mySpeakerName) {
    localStorage.setItem("ai_meet_speaker_name", mySpeakerName);
  } else {
    mySpeakerName = "Khách";
  }
}

// Audio recorder
let audioStream    = null;
let mediaRecorder  = null;
let audioChunks    = [];
let uploadInterval = null;

// ================================================================
// BADGE UI
// ================================================================
const BADGE_CFG = {
  standby:    { color: "#475569", bg: "rgba(71,85,105,0.2)",  icon: "🤖", text: "AI đang lắng nghe..." },
  recording:  { color: "#ef4444", bg: "rgba(239,68,68,0.2)",  icon: "🔴", text: "Đang ghi biên bản",  pulse: true },
  finalizing: { color: "#f59e0b", bg: "rgba(245,158,11,0.2)", icon: "⚡", text: "Đang tạo biên bản..." },
};

function createBadge() {
  if (badgeEl) return;
  badgeEl = document.createElement("div");
  badgeEl.id = "ai-meet-badge";
  document.body.appendChild(badgeEl);
  renderBadge(state);
}

function renderBadge(s) {
  if (!badgeEl) return;
  const c = BADGE_CFG[s] || BADGE_CFG.standby;
  badgeEl.innerHTML = `
    <style>
      #ai-meet-badge-inner {
        position: fixed; bottom: 24px; right: 24px; z-index: 999999;
        background: rgba(5,6,15,0.93); backdrop-filter: blur(16px);
        border: 1px solid ${c.color}44; border-radius: 999px;
        padding: 10px 18px; display: flex; align-items: center; gap: 10px;
        font-family: 'Google Sans', Inter, sans-serif; font-size: 13px;
        font-weight: 600; color: white;
        box-shadow: 0 4px 24px ${c.bg};
        transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
        cursor: default; user-select: none;
      }
      #ai-meet-dot {
        width: 9px; height: 9px; border-radius: 50%;
        background: ${c.color};
        ${c.pulse ? "animation: ai-blink 1.1s ease-in-out infinite;" : ""}
      }
      #ai-meet-count {
        font-size: 11px; color: ${c.color}; font-weight: 700;
        background: ${c.bg}; padding: 2px 8px; border-radius: 99px;
        display: ${transcript.length > 0 ? "block" : "none"};
      }
      @keyframes ai-blink {
        0%,100% { opacity:1; transform:scale(1); }
        50% { opacity:0.25; transform:scale(1.6); }
      }
    </style>
    <div id="ai-meet-badge-inner">
      <div id="ai-meet-dot"></div>
      <span>${c.icon} ${c.text}</span>
      <span id="ai-meet-count">${transcript.length} dòng</span>
    </div>
  `;
}

// ================================================================
// WEB SPEECH API - Nhan dien giong noi tieng Viet TRUC TIEP
// ================================================================
function startSpeechRecognition() {
  if (!("webkitSpeechRecognition" in window)) {
    console.error("[AI] Trinh duyet khong ho tro Speech Recognition");
    return;
  }
  if (isListening) return;

  recognition = new webkitSpeechRecognition();
  recognition.lang        = "vi-VN";   // Tieng Viet
  recognition.continuous  = true;      // Chay lien tuc
  recognition.interimResults = true;   // Hien thi tam thoi
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    isListening = true;
    console.log(`[AI] Speech Recognition bat dau - Room: ${roomId} - State: ${state}`);
  };

  recognition.onend = () => {
    isListening = false;
    // Tu dong khoi dong lai neu chua ket thuc
    if (state !== "finalizing") {
      setTimeout(startSpeechRecognition, 500);
    }
  };

  recognition.onerror = (e) => {
    console.warn("[AI] Loi speech:", e.error);
    isListening = false;
    if (e.error !== "aborted") {
      setTimeout(startSpeechRecognition, 1000);
    }
  };

  recognition.onresult = (event) => {
    // Lay ket qua final (khong phai interim)
    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (!event.results[i].isFinal) continue;

      const text = event.results[i][0].transcript.trim();
      if (!text || text.length < 2) continue;

      console.log(`[AI][${state.toUpperCase()}] Nghe duoc: "${text}"`);
      handleSpeech(text);
    }
  };

  recognition.start();
}

// ================================================================
// XU LY LOGIC TRIGGER + GHI CHU
// ================================================================
async function handleSpeech(text) {
  if (state === "standby") {
    // Kiem tra trigger bat dau
    if (isStartTrigger(text)) {
      await startRecording();
    }
  }
  else if (state === "recording") {
    // Luu dong thoai
    await saveLine(text);

    // Kiem tra trigger ket thuc
    if (isStopTrigger(text)) {
      await endRecording();
    }
  }
}

function isStartTrigger(text) {
  const t = text.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/đ/g,"d");
  const triggers = ["bat dau cuoc hop", "bat dau hop", "bat dau thoi", "bat dau nao",
                    "chung ta bat dau", "minh bat dau", "bat dau ghi"];
  return triggers.some(tr => t.includes(tr));
}

function isStopTrigger(text) {
  const t = text.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/đ/g,"d");
  const triggers = ["ket thuc cuoc hop", "ket thuc hop", "dung ghi", "ket thuc thoi",
                    "dung lai thoi", "het roi", "ket thuc nhe", "dung cuoc hop"];
  return triggers.some(tr => t.includes(tr));
}

async function startRecording() {
  console.log(`[AI] >>> BAT DAU GHI BIEN BAN - Room: ${roomId}`);
  state      = "recording";
  transcript = [];
  renderBadge("recording");

  // Tao meeting trong DB
  try {
    const r = await fetch(`${BACKEND}/start-meeting`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ room_id: roomId })
    });
    const d = await r.json();
    dbMeetingId = d.meeting_id;
    console.log(`[AI] Meeting DB ID: ${dbMeetingId}`);
  } catch(e) {
    console.warn("[AI] Khong luu duoc DB:", e.message);
  }

  // Bat dau ghi am (cho logs/voice)
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(audioStream, { mimeType: "audio/webm" });
    
    mediaRecorder.ondataavailable = e => { 
      if (e.data.size > 0) audioChunks.push(e.data); 
    };
    
    // Yeu cau chunk moi 3 giay de gui lien tuc
    mediaRecorder.start(3000);
    console.log("[AI] Da bat thu am (Voice)");

    // Gui chunk dinh ky len backend
    uploadInterval = setInterval(uploadVoiceChunks, 3500);

  } catch(err) {
    console.warn("[AI] Khong the ghi am micro (ban chua cap quyen?):", err);
  }
}

async function uploadVoiceChunks() {
  if (!audioChunks.length || !dbMeetingId) return;

  const chunksToSend = [...audioChunks];
  audioChunks = []; // reset chua cac chunk tiep theo

  const blob = new Blob(chunksToSend, { type: "audio/webm" });
  const fd = new FormData();
  fd.append("file", blob, `voice_${Date.now()}.webm`);
  fd.append("meeting_id", dbMeetingId);

  try {
    const res = await fetch(`${BACKEND}/upload-voice`, {
      method: "POST",
      body: fd
    });
    console.log("[AI] Da upload voice chunk");
  } catch(e) {
    console.warn("[AI] Loi upload voice:", e);
  }
}

async function saveLine(text) {
  transcript.push({ speaker: mySpeakerName, text });
  renderBadge("recording"); // cap nhat so dong

  // Gui ve backend luu DB + logs/text
  if (dbMeetingId) {
    fetch(`${BACKEND}/add-line`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ meeting_id: dbMeetingId, text, speaker: mySpeakerName })
    }).catch(() => {});
  }

  console.log(`[AI][REC] ${mySpeakerName}: "${text.slice(0,60)}"`);
}

async function endRecording() {
  console.log(`[AI] >>> KET THUC - ${transcript.length} dong`);
  state = "finalizing";
  renderBadge("finalizing");

  // Dung ghi am
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  if (uploadInterval) clearInterval(uploadInterval);
  if (audioStream) audioStream.getTracks().forEach(t => t.stop());
  
  // Nho upload not cac chunk cuoi
  await uploadVoiceChunks();

  // Goi Groq tao bien ban qua backend
  try {
    const r = await fetch(`${BACKEND}/finalize-meeting`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        room_id:    roomId,
        meeting_id: dbMeetingId,
        transcript: transcript
      })
    });
    const d = await r.json();
    console.log("[AI] Bien ban da tao:", d.minutes_path);
    showMinutesNotification("Biên bản đã được tạo thành công!");
  } catch(e) {
    console.error("[AI] Loi finalize:", e);
  }

  // Reset ve standby sau 5s
  setTimeout(() => {
    state       = "standby";
    transcript  = [];
    dbMeetingId = null;
    audioStream = null;
    mediaRecorder = null;
    renderBadge("standby");
  }, 5000);
}

// ================================================================
// THONG BAO BIEN BAN XONG
// ================================================================
function showMinutesNotification(preview) {
  const notif = document.createElement("div");
  notif.style.cssText = `
    position:fixed; bottom:80px; right:24px; z-index:999999;
    background:#064e3b; border:1px solid #059669; border-radius:16px;
    padding:16px 20px; max-width:380px; color:white;
    font-family:'Google Sans',Inter,sans-serif; font-size:13px;
    box-shadow:0 8px 32px rgba(5,150,105,0.4);
    animation: slideUp 0.4s ease;
  `;
  notif.innerHTML = `
    <style>@keyframes slideUp{from{transform:translateY(20px);opacity:0}to{transform:translateY(0);opacity:1}}</style>
    <div style="font-weight:700;margin-bottom:8px">✅ Biên bản đã được tạo!</div>
    <div style="color:#6ee7b7;font-size:12px;line-height:1.5">${preview.slice(0,150)}...</div>
    <div style="margin-top:10px;font-size:11px;color:#6ee7b7">📁 Xem tại: logs/minutes/</div>
  `;
  document.body.appendChild(notif);
  setTimeout(() => notif.remove(), 8000);
}

// ================================================================
// PHAT HIEN PHONG GOOGLE MEET VA KHOI DONG
// ================================================================
function detectRoom() {
  const match = location.href.match(/meet\.google\.com\/([a-z]{3}-[a-z]{4}-[a-z]{3})/);
  if (!match) {
    // Neu roi khoi phong (ve trang thai ngoai phong hop) ma dang recording thi tu dong ngat
    if (state === "recording") {
       endRecording();
    }
    return;
  }
  
  if (badgeEl && roomId === match[1]) return; // Da khoi dong

  roomId = match[1];
  console.log(`[AI] Phong Google Meet: ${roomId}`);
  createBadge();
  startSpeechRecognition();
}

// Chay ngay va theo doi navigation
detectRoom();
new MutationObserver(detectRoom).observe(document.body, { childList: true, subtree: false });

// Tự động lưu và kết thúc nếu người dùng đóng tab / F5
window.addEventListener('beforeunload', () => {
  if (state === "recording") {
    endRecording();
  }
});
