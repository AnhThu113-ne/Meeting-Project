import os
import torch
import numpy as np
from scipy.spatial.distance import cdist
from pydub import AudioSegment
from typing import List, Dict, Any
from groq import Groq

class MeetingAudioProcessor:
    def __init__(self, hf_token: str = None, groq_key: str = None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.groq_client = Groq(api_key=groq_key) if groq_key else None
        
        # 1. Model Diarization (Phân tách người nói)
        try:
            from pyannote.audio import Pipeline, Model, Inference
            self.diarize_pipeline = None
            if hf_token:
                self.diarize_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", token=hf_token
                )
                if torch.cuda.is_available():
                    self.diarize_pipeline.to(self.device)

            # 2. Model Embedding (Để nhận diện giọng nói cụ thể của ai)
            if hf_token:
                self.embedding_model = Model.from_pretrained(
                    "pyannote/embedding", token=hf_token
                )
                self.inference = Inference(self.embedding_model, window="whole")
            else:
                self.inference = None
            self.has_pyannote = True
        except ImportError:
            self.has_pyannote = False
            print("[System] Pyannote Audio không khả dụng. Cần cài đặt pyannote.audio và có HF_TOKEN.")

    def stt_groq(self, segment_audio_path: str):
        """Gửi đoạn âm thanh nhỏ qua Groq Whisper STT"""
        if not self.groq_client: return ""
        try:
            with open(segment_audio_path, 'rb') as f:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(os.path.basename(segment_audio_path), f.read()),
                    model="whisper-large-v3",
                    response_format="json",
                    language="vi"
                )
                return transcription.text
        except Exception as e:
            print(f"[Groq STT Error]: {e}")
            return ""

    def identify_speaker(self, segment_embedding, reference_embeddings: dict):
        """So sánh vân tay giọng nói (Voice Biometrics) bằng thuật toán Cosine Similarity"""
        if segment_embedding is None or not reference_embeddings:
            return "Unknown"
            
        min_dist = 100
        best_name = "Unknown"
        
        for name, ref_emb in reference_embeddings.items():
            if ref_emb is None: continue
            # Tính khoảng cách Cosine
            dist = cdist([segment_embedding], [ref_emb], metric="cosine")[0][0]
            # Mức khoảng cách càng nhỏ càng sát (Threshold thường là ~0.5 - 0.6)
            if dist < 0.6 and dist < min_dist: 
                min_dist = dist
                best_name = name
                
        return best_name

    def process_audio(self, audio_path: str, references: dict = None) -> List[Dict[str, Any]]:
        # Nếu ko cài đặt Pyannote, chạy fall-back chỉ bóc băng
        if not self.has_pyannote or not self.diarize_pipeline:
            print("[Warning] Không có thư viện Pyannote. Audio sẽ được xử lý STT nguyên khối.")
            text = self.stt_groq(audio_path)
            return [{"speaker": "Unknown", "text": text, "start": 0.0, "end": 0.0}] if text else []

        print(f"[AI] Bắt đầu trích xuất nhãn vân tay giọng từ Database ({len(references) if references else 0} người mẫu)...")
        ref_embs = {}
        if references and self.inference:
            for name, path in references.items():
                if os.path.exists(path):
                    try:
                       ref_embs[name] = self.inference(path)
                    except:
                       pass

        print("[AI] Diarization: Bắt đầu phân tách sóng âm thanh...")
        diarization = self.diarize_pipeline(audio_path)
        
        print("[AI] Tách xuất tín hiệu Audio thô để nhận dạng văn bản (STT)...")
        audio = AudioSegment.from_file(audio_path)
        
        final_transcript = []
        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            chunk = audio[start_ms:end_ms]
            
            # Lưu tạm chunk thành file .wav cho AI Groq và Embedding Model
            temp_chunk_path = f"temp_chunk_{speaker_id}_{start_ms}.wav"
            chunk.export(temp_chunk_path, format="wav")
            
            # Trích xuất vân tay âm thanh và Định danh
            seg_emb = self.inference(temp_chunk_path) if self.inference else None
            real_name = self.identify_speaker(seg_emb, ref_embs)
            
            # Nếu nhận dạng đúng khớp mẫu voice, đổi tên. Ngược lại để nhãn mặc định.
            display_name = real_name if real_name != "Unknown" else f"Speaker_{speaker_id}"
            
            # Bóc chữ đoạn hội thoại
            text = self.stt_groq(temp_chunk_path)
            
            if text and text.strip():
                final_transcript.append({
                    "speaker": display_name,
                    "text": text.strip(),
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2)
                })
                print(f"[{display_name}] {text.strip()}")
            
            if os.path.exists(temp_chunk_path):
                os.remove(temp_chunk_path)
                
        return final_transcript
