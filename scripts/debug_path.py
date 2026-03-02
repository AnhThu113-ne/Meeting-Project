try:
    import whisper
    print(f"Whisper loaded from: {whisper.__file__}")
except Exception as e:
    print(f"Error loading whisper: {e}")
    import traceback
    traceback.print_exc()
