import os
import torch
import numpy as np
from scipy.spatial.distance import cdist
from pydub import AudioSegment
from typing import List, Dict, Any
from groq import Groq

class MeetingAudioProcessor:
    def __init__(self, hf_token: str = None, groq_key: str = None, viettel_key: str = None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        api_key = groq_key or viettel_key or os.getenv("GROQ_API_KEY", "")
        self.groq_client = Groq(api_key=api_key) if api_key else None
        
        # Khởi tạo mô hình Faster-Whisper offline nếu không có Groq API Key
        self.whisper_model = None
        if not self.groq_client:
            try:
                from faster_whisper import WhisperModel
                print("[AI STT] Khoi tao mo hinh Faster-Whisper base (Offline)...")
                # Dùng base hoặc tiny để chạy nhẹ trên CPU
                self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
                print("[AI STT] Khoi tao Faster-Whisper thanh cong!")
            except Exception as e:
                print(f"[AI STT Error] Khong the khoi tao Faster-Whisper: {e}")
        
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
        """Gửi đoạn âm thanh nhỏ qua Groq Whisper STT. Nếu lỗi hoặc không có key, chuyển sang Faster-Whisper offline."""
        if self.groq_client:
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
                print(f"[Groq STT Error] Loi khi goi Groq API: {e}. Chuyen sang Faster-Whisper offline...")
        
        # Fallback offline dùng Faster-Whisper
        if self.whisper_model:
            try:
                segments, info = self.whisper_model.transcribe(segment_audio_path, beam_size=5, language="vi")
                text = "".join(segment.text for segment in segments)
                return text
            except Exception as e:
                print(f"[Offline STT Error] Loi transcribing: {e}")
                
        return ""

    def stt_viettel(self, chunk_path: str):
        """Mô phỏng Viettel STT bằng cách sử dụng Groq Whisper STT"""
        return self.stt_groq(chunk_path)

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

    def extract_voice_print_local(self, file_path: str) -> np.ndarray:
        """Trích xuất vân tay giọng nói (MFCC vector) dùng librosa & pydub."""
        temp_wav_path = None
        try:
            import librosa
            import uuid
            # Đọc audio bằng pydub trước để hỗ trợ nhiều định dạng (webm, ogg...)
            sound = AudioSegment.from_file(file_path)
            sound = sound.set_frame_rate(16000).set_channels(1)
            
            # Lưu tạm ra wav tiêu chuẩn
            temp_wav_path = f"temp_vp_{uuid.uuid4().hex[:6]}.wav"
            sound.export(temp_wav_path, format="wav")
            
            # Load bằng librosa
            y, sr = librosa.load(temp_wav_path, sr=16000)
            if len(y) == 0:
                return None
                
            # Cắt bớt phần lặng ở đầu/cuối
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            if len(y_trimmed) == 0:
                y_trimmed = y
                
            # Trích xuất MFCC (20 hệ số)
            mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=20)
            
            # Loại bỏ hệ số MFCC-0 (năng lượng tổng thể) vì nó chiếm biên độ quá lớn
            # và gây nhiễu khi tính cosine distance giữa các người nói
            mfccs = mfccs[1:]
            
            # Tính trung bình và độ lệch chuẩn theo thời gian
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Chuẩn hóa từng hệ số bằng độ lệch chuẩn của chính nó (tỉ lệ mean/std)
            # giúp phân bổ trọng số đồng đều, tạo khoảng cách phân biệt lớn nhất giữa các giọng
            vector = mfcc_mean / (mfcc_std + 1e-6)
            
            # Chuẩn hóa L2 vector đặc trưng
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            return vector
        except Exception as e:
            print(f"[AI Local] Loi trich xuat van tay giong tu {file_path}: {e}")
            return None
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except:
                    pass

    def process_audio_local(self, audio_path: str, references: dict = None) -> List[Dict[str, Any]]:
        """Phân tách và nhận dạng giọng nói ngoại tuyến hoàn toàn bằng khoảng lặng + MFCC."""
        print("[AI Local] Dang tai file audio cuộc họp...")
        try:
            sound = AudioSegment.from_file(audio_path)
        except Exception as e:
            print(f"[AI Local Error] Khong the doc file audio {audio_path}: {e}")
            return []

        # Chuyển đổi references thành MFCC vectors
        ref_vectors = {}
        if references:
            for name, path in references.items():
                if os.path.exists(path):
                    vec = self.extract_voice_print_local(path)
                    if vec is not None:
                        ref_vectors[name] = vec
            print(f"[AI Local] Da tai {len(ref_vectors)} mau giong noi: {list(ref_vectors.keys())}")

        # Phân đoạn dựa trên khoảng lặng (min_silence_len=1000ms, silence_thresh=-40dB)
        print("[AI Local] Dang phan tach phan doan noi dua tren khoang lang...")
        from pydub.silence import detect_nonsilent
        non_silent_ranges = detect_nonsilent(
            sound,
            min_silence_len=1000,
            silence_thresh=-40,
            seek_step=10
        )

        # Thêm biên đệm (padding) 500ms vào đầu và cuối mỗi phân đoạn để tránh mất chữ ở biên
        padded_ranges = []
        for start_ms, end_ms in non_silent_ranges:
            padded_start = max(0, start_ms - 500)
            padded_end = min(len(sound), end_ms + 500)
            padded_ranges.append([padded_start, padded_end])

        # Sắp xếp và gộp các phân đoạn bị đè lên nhau hoặc quá sát nhau (khoảng cách <= 200ms)
        merged_ranges = []
        if padded_ranges:
            padded_ranges.sort(key=lambda x: x[0])
            current_start, current_end = padded_ranges[0]
            for next_start, next_end in padded_ranges[1:]:
                if next_start <= current_end + 200:
                    current_end = max(current_end, next_end)
                else:
                    merged_ranges.append([current_start, current_end])
                    current_start, current_end = next_start, next_end
            merged_ranges.append([current_start, current_end])
        else:
            merged_ranges = [[0, len(sound)]]

        final_transcript = []
        speaker_counter = 1
        cluster_vectors = [] # list of (speaker_label, vector)

        for idx, (start_ms, end_ms) in enumerate(merged_ranges):
            duration_ms = end_ms - start_ms
            if duration_ms < 500: # Bỏ qua đoạn quá ngắn
                continue

            # Cắt phân đoạn âm thanh và đảm bảo định dạng 16kHz mono chuẩn
            chunk = sound[start_ms:end_ms]
            chunk = chunk.set_frame_rate(16000).set_channels(1)
            
            # Xuất ra file wav tạm thời để xử lý
            import uuid
            temp_chunk_path = f"temp_local_chunk_{idx}_{uuid.uuid4().hex[:4]}.wav"
            chunk.export(temp_chunk_path, format="wav")

            # 1. Trích xuất MFCC vector cho segment này
            seg_vec = self.extract_voice_print_local(temp_chunk_path)
            
            # 2. Nhận diện người nói
            speaker_name = "Unknown"
            if seg_vec is not None:
                # 2a. Đối chiếu với các giọng nói đã đăng ký
                min_dist = 100.0
                best_match = None
                
                print(f"[AI Local] --- Đang đối chiếu phân đoạn {idx} ({start_ms/1000.0}s - {end_ms/1000.0}s) ---")
                for ref_name, ref_vec in ref_vectors.items():
                    dist = cdist([seg_vec], [ref_vec], metric="cosine")[0][0]
                    print(f"  + So sanh voi '{ref_name}': cosine distance = {dist:.4f}")
                    if dist < min_dist:
                        min_dist = dist
                        best_match = ref_name
                
                # Ngưỡng Cosine distance khớp giọng mẫu: <= 0.16 là khớp tốt (dựa trên thực nghiệm gap 0.10 - 0.19)
                if best_match and min_dist < 0.16:
                    speaker_name = best_match
                    print(f"  => KHOP mau dang ky: '{speaker_name}' (khoang cach: {min_dist:.4f})")
                else:
                    # 2b. Gom cụm tự động (Clustering) khi không khớp ai đã đăng ký
                    best_cluster = None
                    min_cluster_dist = 100.0
                    for cluster_label, c_vec in cluster_vectors:
                        dist = cdist([seg_vec], [c_vec], metric="cosine")[0][0]
                        if dist < min_cluster_dist:
                            min_cluster_dist = dist
                            best_cluster = cluster_label
                    
                    if best_cluster and min_cluster_dist < 0.16:
                        speaker_name = best_cluster
                        print(f"  => KHOP cum tu dong: '{speaker_name}' (khoang cach: {min_cluster_dist:.4f})")
                        # Cập nhật vector của cụm (trung bình động)
                        for c_idx, (lbl, c_vec) in enumerate(cluster_vectors):
                            if lbl == best_cluster:
                                cluster_vectors[c_idx] = (lbl, 0.8 * c_vec + 0.2 * seg_vec)
                                break
                    else:
                        speaker_name = f"Người nói {speaker_counter}"
                        speaker_counter += 1
                        cluster_vectors.append((speaker_name, seg_vec))
                        print(f"  => TAO cum moi: '{speaker_name}'")
            
            # 3. Chuyển đổi giọng nói thành văn bản (STT)
            text = self.stt_groq(temp_chunk_path)
            
            if os.path.exists(temp_chunk_path):
                try:
                    os.remove(temp_chunk_path)
                except:
                    pass

            if text and text.strip():
                final_transcript.append({
                    "speaker": speaker_name,
                    "text": text.strip(),
                    "start": round(start_ms / 1000.0, 2),
                    "end": round(end_ms / 1000.0, 2)
                })
                print(f"[AI Local] [{speaker_name}] ({start_ms/1000.0}s - {end_ms/1000.0}s): {text.strip()}")

        return final_transcript

    def process_audio(self, audio_path: str, references: dict = None) -> List[Dict[str, Any]]:
        # Nếu ko cài đặt Pyannote hoặc thiếu HF_TOKEN, chạy bộ nhận diện cục bộ (Local Biometrics)
        if not self.has_pyannote or not self.diarize_pipeline:
            print("[Warning] Khong co thu vien Pyannote hoac thieu HF_TOKEN. Su dung bo xu ly offline cuc bo.")
            return self.process_audio_local(audio_path, references)

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
