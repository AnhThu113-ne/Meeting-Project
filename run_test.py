import requests, time, sys

BASE = "http://localhost:8000"
ROOM = "hop-final-test"

def speak(text):
    r = requests.post(f"{BASE}/test-speak", data={"room_id": ROOM, "text": text})
    d = r.json()
    print(f"  [{d.get('event', d.get('state'))}] {text[:50]}")
    return d

print("--- BAT DAU ---")
speak("bat dau cuoc hop")
time.sleep(0.5)
speak("Em da hoan thanh module nhan dien giong noi")
speak("Thay se review tuan sau")
speak("Demo san pham ngay 15 thang 3")
print("--- KET THUC ---")
speak("ket thuc cuoc hop")

print("Cho Gemini xu ly (toi da 3 phut)...")
for i in range(45):
    time.sleep(4)
    r = requests.get(f"{BASE}/result/{ROOM}")
    data = r.json()
    if "minutes" in data:
        print()
        print("=== BIEN BAN GEMINI ===")
        print(data["minutes"][:1000])
        print("======================")
        sys.exit(0)
    if "error" in data:
        print(f"LOI: {data['error']}")
        sys.exit(1)
    print(f"  [{i*4}s] chua san sang...")

print("TIMEOUT - Xem log backend de biet them")
