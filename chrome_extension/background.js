// background.js - Service Worker
// TU DONG LAT LANG KHI VAO GOOGLE MEET
// Goi Viettel STT moi 5 giay, phat hien "bat dau cuoc hop" de ghi log

const BACKEND_URL  = "http://localhost:8000";
const CHUNK_MS     = 5000;   // 5 giay / chunk

let mediaRecorder   = null;
let audioChunks     = [];
let roomId          = null;
let isCapturing     = false;
let chunkInterval   = null;
let lastRoomState   = "standby";

// -------------------------------------------------------
// Lang nghe message tu popup.js va content.js
// -------------------------------------------------------
chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg.action === "START_STANDBY") {
    roomId = msg.roomId;
    startCapture(msg.tabId);
    reply({ status: "started", roomId });
  }
  if (msg.action === "STOP_CAPTURE") {
    stopCapture();
    reply({ status: "stopped" });
  }
  if (msg.action === "GET_STATUS") {
    reply({ isCapturing, roomId, roomState: lastRoomState });
  }
  if (msg.action === "ROOM_STATE_UPDATE") {
    // Broadcast trang thai moi den tat ca content scripts
    broadcastState(msg.roomState);
  }
  return true;
});

// -------------------------------------------------------
// Bat dau bat audio tu tab Google Meet
// -------------------------------------------------------
async function startCapture(tabId) {
  if (isCapturing) return;

  try {
    const stream = await chrome.tabCapture.capture({
      audio: true,
      video: false,
      audioConstraints: {
        mandatory: { echoCancellation: true, noiseSuppression: true }
      }
    });

    if (!stream) {
      console.error("[AI] Không thể bắt audio.");
      return;
    }

    isCapturing  = true;
    audioChunks  = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.start(CHUNK_MS);

    // Cu 5 giay, lay chunk moi nhat va gui len backend
    chunkInterval = setInterval(sendCurrentChunk, CHUNK_MS);
    console.log(`[AI] Standby mode - Room: ${roomId}`);

  } catch (err) {
    console.error("[AI] Loi capture:", err);
    isCapturing = false;
  }
}

async function sendCurrentChunk() {
  if (!audioChunks.length || !roomId) return;

  const chunks = [...audioChunks];
  audioChunks  = [];

  const blob     = new Blob(chunks, { type: "audio/webm" });
  const formData = new FormData();
  formData.append("file", blob, `chunk_${Date.now()}.webm`);
  formData.append("room_id", roomId);

  try {
    const res  = await fetch(`${BACKEND_URL}/upload-chunk`, { method: "POST", body: formData });
    const data = await res.json();
    const newState = data.room_state || "standby";

    if (newState !== lastRoomState) {
      lastRoomState = newState;
      console.log(`[AI] Trang thai phong: ${newState}`);
      broadcastState(newState);
    }
  } catch (err) {
    console.warn("[AI] Backend chua san sang:", err.message);
  }
}

function stopCapture() {
  if (mediaRecorder && isCapturing) {
    mediaRecorder.stop();
    clearInterval(chunkInterval);
    isCapturing    = false;
    lastRoomState  = "standby";
    console.log("[AI] Da dung capture.");
  }
}

// Gui trang thai moi ve content.js de cap nhat badge
function broadcastState(state) {
  chrome.tabs.query({ url: "https://meet.google.com/*" }, (tabs) => {
    tabs.forEach(tab => {
      chrome.tabs.sendMessage(tab.id, { action: "UPDATE_BADGE", state });
    });
  });
}

