"""
trigger_detector.py
Phát hiện từ khóa kích hoạt / dừng cuộc họp trong văn bản STT.
"""

# Từ khóa bắt đầu cuộc họp (nhận nhiều biến thể)
START_TRIGGERS = [
    "bắt đầu cuộc họp",
    "bat dau cuoc hop",
    "bắt đầu họp",
    "bat dau hop",
    "bắt đầu ghi",
    "bắt đầu thôi",
    "bắt đầu nào",
    "bắt đầu cuộc họp nhé",
    "chúng ta bắt đầu",
    "mình bắt đầu nhé",
]

# Từ khóa kết thúc cuộc họp
STOP_TRIGGERS = [
    "kết thúc cuộc họp",
    "ket thuc cuoc hop",
    "kết thúc họp",
    "kết thúc thôi",
    "họp kết thúc",
    "dừng ghi",
    "dừng cuộc họp",
    "kết thúc nào",
    "tạm dừng",
    "hết rồi",
    "kết thúc nhé",
    "chúng ta dừng ở đây",
    "dừng lại",
]


def normalize_text(text: str) -> str:
    """Chuẩn hóa text: lowercase, bỏ dấu câu thừa."""
    import unicodedata, re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # bỏ dấu câu
    return text


def check_start_trigger(text: str) -> bool:
    """Kiểm tra xem text có chứa từ khóa bắt đầu không."""
    normalized = normalize_text(text)
    for trigger in START_TRIGGERS:
        if trigger in normalized:
            print(f"[TRIGGER] Phat hien tu khoa BAT DAU: '{trigger}' trong: '{text[:50]}'")
            return True
    return False


def check_stop_trigger(text: str) -> bool:
    """Kiểm tra xem text có chứa từ khóa kết thúc không."""
    normalized = normalize_text(text)
    for trigger in STOP_TRIGGERS:
        if trigger in normalized:
            print(f"[TRIGGER] Phat hien tu khoa KET THUC: '{trigger}' trong: '{text[:50]}'")
            return True
    return False
