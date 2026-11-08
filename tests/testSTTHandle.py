import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handlers.STTHandle import (
    STTError,
    STTHandle,
)


def main() -> None:
    stt = STTHandle()
    try:
        input("Naciśnij Enter, aby rozpocząć nagrywanie...")
        stt.start_recording()
        print("Nagrywanie trwa. Naciśnij Enter, aby je zakończyć.")
        input()
        transcription = stt.stop_recording()
        print(f"\nTranskrypcja: {transcription}")
    except STTError as exc:
        print(f"Błąd STT: {exc}")


if __name__ == "__main__":
    main()