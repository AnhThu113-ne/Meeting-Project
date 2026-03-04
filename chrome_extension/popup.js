// popup.js - Điều khiển giao diện popup Extension

const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const statusSub = document.getElementById("statusSub");
const meetingIdBox = document.getElementById("meetingIdBox");
const meetingIdText = document.getElementById("meetingIdText");
const transcriptPreview = document.getElementById("transcriptPreview");
const btnStart = document.getElementById("btnStart");
const btnStop = document.getElementById("btnStop");
const btnDashboard = document.getElementById("btnDashboard");

const BACKEND_URL = "http://localhost:8000";
let currentMeetingId = null;
let pollInterval = null;

// --- Khởi tạo: Kiểm tra trạng thái hiện tại ---
chrome.runtime.sendMessage({ action: "GET_STATUS" }, (response) => {
  if (response?.isRecording) {
    setRecordingState(true, response.meetingId);
  }
});

// --- Nút bắt đầu ghi ---
btnStart.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab.url.includes("meet.google.com")) {
    statusLabel.textContent = "⚠️ Hãy mở Google Meet trước!";
    statusSub.textContent = "Chuyển đến tab Google Meet của bạn.";
    return;
  }

  chrome.runtime.sendMessage(
    { action: "START_RECORDING", tabId: tab.id, meetingId: extractMeetId(tab.url) },
    (response) => {
      if (response?.status === "started") {
        const meetId = extractMeetId(tab.url);
        currentMeetingId = meetId;
        setRecordingState(true, meetId);
        startPollingTranscript(meetId);
      }
    }
  );
});

// --- Nút dừng ghi ---
btnStop.addEventListener("click", () => {
  chrome.runtime.sendMessage({ action: "STOP_RECORDING" }, () => {
    setRecordingState(false);
    clearInterval(pollInterval);
    statusLabel.textContent = "✅ Đang tạo biên bản...";
    statusSub.textContent = "Gemini AI đang xử lý. Mở Dashboard để xem.";
    statusDot.className = "status-dot processing";
  });
});

// --- Mở Dashboard ---
btnDashboard.addEventListener("click", () => {
  chrome.tabs.create({ url: "http://localhost:5173" });
});

// --- Cập nhật giao diện ---
function setRecordingState(recording, meetId = null) {
  if (recording) {
    btnStart.style.display = "none";
    btnStop.style.display = "block";
    statusDot.className = "status-dot recording";
    statusLabel.textContent = "🔴 Đang ghi âm";
    statusSub.textContent = "AI đang lắng nghe và nhận diện giọng nói...";
    meetingIdBox.className = "meeting-id-box visible";
    meetingIdText.textContent = meetId || "—";
  } else {
    btnStart.style.display = "block";
    btnStop.style.display = "none";
  }
}

// --- Polling để lấy preview transcript ---
function startPollingTranscript(meetId) {
  pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/live-transcript/${meetId}`);
      const data = await res.json();
      if (data.lines && data.lines.length > 0) {
        renderTranscript(data.lines.slice(-5)); // Hiện 5 dòng cuối
      }
    } catch (e) { /* Backend chưa sẵn sàng */ }
  }, 3000);
}

function renderTranscript(lines) {
  transcriptPreview.innerHTML = lines.map((l, i) => {
    const colorClass = i % 2 === 0 ? "spk-a" : "spk-b";
    return `<div class="line"><span class="spk ${colorClass}">${l.speaker}:</span>${l.text}</div>`;
  }).join("");
}

function extractMeetId(url) {
  const match = url.match(/meet\.google\.com\/([a-z]{3}-[a-z]{4}-[a-z]{3})/);
  return match ? match[1] : `meeting_${Date.now()}`;
}
