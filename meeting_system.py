import os
import base64
import requests
import numpy as np
import sounddevice as sd 
from scipy.io.wavfile import write
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
from speechbrain.inference.classifiers import EncoderClassifier 
from pyannote.audio import Pipeline 
import google.generativeai as genai  
from torchaudio.transforms import MFCC  
import torchaudio

# Replace with your API keys
VIETTEL_API_TOKEN = "your_viettel_api_token"
GEMINI_API_KEY = "your_gemini_api_key"
PYANNOTE_TOKEN = "your_huggingface_token"

# Viettel API endpoints (based on common knowledge, verify with documentation)
VIETTEL_STT_ENDPOINT = "https://viettelgroup.ai/voice/api/asr/v1/rest/decode"
VIETTEL_TTS_ENDPOINT = "https://viettelgroup.ai/voice/api/tts/v1/rest/syn"

# Enrolled speakers: dict of name to embedding
enrolled_speakers = {}

# Load models
speaker_recognition = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")  # Thay SpeakerRecognition bằng EncoderClassifier
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=PYANNOTE_TOKEN)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Function to extract MFCC and embedding
def extract_embedding(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    mfcc_transform = MFCC(sample_rate=sample_rate)
    mfccs = mfcc_transform(waveform)
    # Average MFCC for simple feature, but use embedding
    embedding = speaker_recognition.encode_batch(waveform)
    return embedding.squeeze()

# Function to enroll speaker
def enroll_speaker(name, audio_path):
    embedding = extract_embedding(audio_path)
    enrolled_speakers[name] = embedding
    messagebox.showinfo("Enrollment", f"Enrolled {name} successfully.")

# Function to identify speaker from embedding
def identify_speaker(embedding):
    max_sim = 0
    identified = "Unknown"
    for name, emb in enrolled_speakers.items():
        sim = speaker_recognition.similarity(emb, embedding) 
        if sim > max_sim:
            max_sim = sim
            identified = name
    if max_sim > 0.7:  # Threshold
        return identified
    return "Unknown"

# Function to record audio chunk
def record_audio(duration=5, fs=16000):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        write(tmp.name, fs, recording)
        return tmp.name

# Function to STT using Viettel
def stt(audio_path):
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    headers = {"token": VIETTEL_API_TOKEN}
    files = {"file": (os.path.basename(audio_path), audio_data, "audio/wav")}
    response = requests.post(VIETTEL_STT_ENDPOINT, headers=headers, files=files)
    if response.status_code == 200:
        return response.json().get("result", "")
    return ""

# Function to TTS using Viettel
def tts(text, output_path="output.wav"):
    headers = {"token": VIETTEL_API_TOKEN, "Content-Type": "application/json"}
    data = {
        "text": text,
        "voice": "hn-female",  
        "speed": 1.0,
        "tts_return_option": "2"  
    }
    response = requests.post(VIETTEL_TTS_ENDPOINT, headers=headers, json=data)
    if response.status_code == 200:
        audio_base64 = response.json().get("audio_content")
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(audio_base64))
        os.system(f"aplay {output_path}")  
    else:
        print("TTS failed")

# Function to process meeting audio
def process_meeting(audio_path):
    # Diarization
    diarization = diarization_pipeline(audio_path)
    transcript = ""
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        text = stt(audio_path)  # Better to extract segment and stt
        identified = "Speaker " + speaker  # Use clustering label
        transcript += f"{identified}: {text}\n"
    return transcript

# Function to generate minutes
def generate_minutes(transcript):
    prompt = f"Tạo biên bản cuộc họp từ bản ghi sau: {transcript}"
    response = model.generate_content(prompt)
    return response.text

# Tkinter UI
root = tk.Tk()
root.title("AI Meeting Minutes System")
root.geometry("600x400")

# Enroll button
def enroll_button():
    name = entry_name.get()
    file = filedialog.askopenfilename(title="Select Audio File (15-30 min)", filetypes=(("wav files", "*.wav"),))
    if file:
        enroll_speaker(name, file)

label_name = tk.Label(root, text="Speaker Name:")
entry_name = tk.Entry(root)
btn_enroll = tk.Button(root, text="Enroll Speaker", command=enroll_button)

# Start listening
def start_listening():
    while True:
        audio_path = record_audio(duration=10)  # Listen in chunks
        text = stt(audio_path)
        embedding = extract_embedding(audio_path)
        name = identify_speaker(embedding)
        if "chào" in text.lower():  # Greeting
            response = f"Chào {name}"
            tts(response)
        if "bắt đầu cuộc họp" in text.lower():
            messagebox.showinfo("Meeting", "Bắt đầu ghi cuộc họp.")
            # Start recording full meeting
            meeting_path = "meeting.wav"
            # Record until stop, but for prototype, assume upload
            break
        os.unlink(audio_path)
    # After start, record meeting
    # For prototype, ask upload meeting audio
    meeting_file = filedialog.askopenfilename(title="Upload Meeting Audio")
    if meeting_file:
        transcript = process_meeting(meeting_file)
        minutes = generate_minutes(transcript)
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, minutes)

btn_start = tk.Button(root, text="Start System", command=start_listening)

text_area = tk.Text(root, height=20)

# Pack
label_name.pack()
entry_name.pack()
btn_enroll.pack()
btn_start.pack()
text_area.pack()

root.mainloop()