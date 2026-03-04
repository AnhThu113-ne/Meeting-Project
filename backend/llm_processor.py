"""
llm_processor.py
Hỗ trợ 3 backend tạo biên bản:
1. Groq  (Llama 3.3 70B - Free, 6000 RPM) ← Ưu tiên
2. Gemini (gemini-2.0-flash)
3. Rule-based (offline, không cần API)
"""
import os
import re
from datetime import datetime


class MeetingLLMProcessor:
    def __init__(self, api_key: str = None, groq_key: str = None):
        self.gemini_key = api_key
        self.groq_key   = groq_key or os.getenv("GROQ_API_KEY", "")

        # Chon backend
        if self.groq_key:
            self.backend = "groq"
            self._init_groq()
        elif self.gemini_key:
            self.backend = "gemini"
            self._init_gemini()
        else:
            self.backend = "rule"
        print(f"[LLM] Su dung backend: {self.backend.upper()}")

    def _init_groq(self):
        from groq import Groq
        self.groq_client = Groq(api_key=self.groq_key)

    def _init_gemini(self):
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_key)
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")

    # ------------------------------------------------------------------
    def generate_minutes(self, dialog_transcript: list) -> str:
        if self.backend == "groq":
            try:
                return self._groq_minutes(dialog_transcript)
            except Exception as e:
                print(f"[LLM] Groq loi: {e} → fallback rule-based")
        elif self.backend == "gemini":
            try:
                return self._gemini_minutes(dialog_transcript)
            except Exception as e:
                print(f"[LLM] Gemini loi: {e} → fallback rule-based")

        return self._rule_minutes(dialog_transcript)

    # ------------------------------------------------------------------
    # GROQ - Llama 3.3 70B
    # ------------------------------------------------------------------
    def _groq_minutes(self, transcript: list) -> str:
        text = self._format_transcript(transcript)
        chat = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là thư ký cuộc họp chuyên nghiệp. "
                        "Hãy tạo biên bản cuộc họp chính thức bằng tiếng Việt, "
                        "định dạng Markdown rõ ràng."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Tạo biên bản cuộc họp từ transcript sau:\n\n{text}\n\n"
                        "Biên bản gồm:\n"
                        "1. Tiêu đề cuộc họp\n"
                        "2. Thời gian & thành phần tham dự\n"
                        "3. Tóm tắt nội dung chính\n"
                        "4. Quyết định quan trọng\n"
                        "5. Action Items (ai làm gì, deadline)\n"
                        "6. Kết luận"
                    )
                }
            ],
            temperature=0.3,
            max_tokens=2048
        )
        return chat.choices[0].message.content

    # ------------------------------------------------------------------
    # GEMINI
    # ------------------------------------------------------------------
    def _gemini_minutes(self, transcript: list) -> str:
        import time
        text = self._format_transcript(transcript)
        prompt = (
            "Tạo biên bản cuộc họp chính thức bằng tiếng Việt từ transcript sau.\n"
            "Định dạng Markdown, gồm: tiêu đề, tóm tắt, quyết định, action items, kết luận.\n\n"
            f"TRANSCRIPT:\n{text}"
        )
        for delay in [5, 20, 60]:
            try:
                res = self.gemini_model.generate_content(prompt)
                return res.text
            except Exception as e:
                if any(k in str(e).lower() for k in ["quota", "rate", "429", "exhausted"]):
                    print(f"[Gemini] Rate limit, cho {delay}s...")
                    time.sleep(delay)
                else:
                    raise
        res = self.gemini_model.generate_content(prompt)
        return res.text

    # ------------------------------------------------------------------
    # RULE-BASED (offline, không cần AI)
    # ------------------------------------------------------------------
    def _rule_minutes(self, transcript: list) -> str:
        now   = datetime.now().strftime("%d/%m/%Y %H:%M")
        lines = [f"- **{t.get('speaker','?')}**: {t.get('text','')}" for t in transcript]

        # Đếm số từ mỗi người nói
        speakers = {}
        for t in transcript:
            spk = t.get("speaker", "Unknown")
            speakers[spk] = speakers.get(spk, 0) + len(t.get("text", "").split())

        attendees = ", ".join(f"**{s}**" for s in speakers.keys()) or "Chưa xác định"

        md = f"""# 📋 BIÊN BẢN CUỘC HỌP

**Thời gian:** {now}
**Thành phần tham dự:** {attendees}

---

## 1. Nội dung cuộc họp

{chr(10).join(lines)}

---

## 2. Thống kê phát biểu

| Người nói | Số từ |
|-----------|-------|
"""
        for spk, cnt in speakers.items():
            md += f"| {spk} | {cnt} |\n"

        md += f"""
---

## 3. Tóm tắt

Cuộc họp có **{len(transcript)} lượt phát biểu** từ **{len(speakers)} người tham gia**.

> ⚠️ *Biên bản này được tạo tự động (không dùng AI). Vui lòng bổ sung thêm thông tin.*

---

## 4. Action Items

- [ ] _(Cần bổ sung thủ công)_

---

## 5. Kết luận

Cuộc họp kết thúc lúc **{now}**.
"""
        return md

    # ------------------------------------------------------------------
    def _format_transcript(self, transcript: list) -> str:
        return "\n".join(
            f"{t.get('speaker','?')}: {t.get('text','')}"
            for t in transcript
        )
