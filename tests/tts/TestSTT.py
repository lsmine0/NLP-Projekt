import time
import os
import site
from pathlib import Path

# Automatyczne dodanie ścieżek bibliotek NVIDIA CUDA z venv na systemie Windows
for sp in site.getsitepackages():
    for lib in ["cublas", "cudnn"]:
        dll_path = os.path.join(sp, "nvidia", lib, "bin")
        if os.path.exists(dll_path):
            os.add_dll_directory(dll_path)
            os.environ["PATH"] = dll_path + os.pathsep + os.environ["PATH"]

import ctranslate2
from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "tts_outputs"


def test_stt(audio_path: Path, output_txt_path: Path = BASE_DIR / "stt_wynik.txt", model_size: str = "base", device: str = None):
    if device is None:
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"--- Ładowanie modelu Faster-Whisper ({model_size}) na {device} ({compute_type}) ---")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    start_time = time.perf_counter()
    segments, info = model.transcribe(str(audio_path), beam_size=5)

    lines_with_timestamps = []
    full_text = []

    for segment in segments:
        line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}"
        print(line)
        lines_with_timestamps.append(line)
        full_text.append(segment.text.strip())

    elapsed = time.perf_counter() - start_time
    rtf = elapsed / info.duration if info.duration > 0 else 0

    summary = (
        f"\n--- PODSUMOWANIE STT ---\n"
        f"Plik wejściowy: {audio_path.name}\n"
        f"Model: {model_size} ({device}, {compute_type})\n"
        f"Wykryty język: {info.language} (pewność: {info.language_probability:.2f})\n"
        f"Czas transkrypcji: {elapsed:.2f}s | Długość audio: {info.duration:.2f}s\n"
        f"RTF (Real-Time Factor): {rtf:.3f}\n"
    )
    print(summary)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("PEŁNY TEKST:\n")
        f.write(" ".join(full_text) + "\n\n")
        f.write("SEGMENTY Z CZASEM:\n")
        f.write("\n".join(lines_with_timestamps) + "\n")
        f.write(summary)

    print(f"Zapisano wynik transkrypcji do pliku: {output_txt_path}")
    return " ".join(full_text)


if __name__ == "__main__":
    # Szuka pierwszego dostępnego pliku audio z testów TTS lub ref_voice.wav
    sample_candidates = [
        OUTPUT_DIR / "xtts.wav",
        OUTPUT_DIR / "chatterbox.wav",
        OUTPUT_DIR / "piper.wav",
        OUTPUT_DIR / "kokoro.wav",
        BASE_DIR / "ref_voice.wav",
    ]
    target_audio = next((p for p in sample_candidates if p.exists()), None)

    if target_audio:
        test_stt(target_audio, output_txt_path=BASE_DIR / "stt_wynik.txt")
    else:
        print("Brak pliku .wav do przetestowania! Uruchom najpierw TestTTS.py.")