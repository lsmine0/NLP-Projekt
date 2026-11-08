import os
import time
import wave
import re
import urllib.request
import warnings
from pathlib import Path
import numpy as np

# Wyciszenie niepotrzebnych ostrzeżeń w konsoli
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ścieżki względne
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "tts_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Tekst testowy (możesz też wczytywać z pliku hamlet.txt)
DEFAULT_TEXT_PL = "Cześć! To jest test porównawczy systemów zamiany tekstu na mowę. Sprawdzamy jak poszczególne modele radzą sobie z językiem polskim."
DEFAULT_TEXT_EN = "Hello! This is a benchmark test of text to speech synthesis models."

def load_text_safely(filepath: Path, fallback_text: str) -> str:
    """Wczytuje plik tekstowy automatycznie wykrywając UTF-8 lub Windows-1250."""
    if not filepath.exists():
        print(f"[Info] Brak pliku {filepath.name}, używam tekstu domyślnego.")
        return fallback_text
    for enc in ["utf-8-sig", "utf-8", "cp1250", "iso-8859-2"]:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Nie udało się rozpoznać kodowania pliku: {filepath}")


def split_text_into_chunks(text: str, max_chars: int = 200) -> list[str]:
    """Dzieli tekst na fragmenty gwarantując, że żaden nie przekroczy max_chars."""
    clean_text = text.replace("\r\n", " ").replace("\n", " ").replace("--", ", ")
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    
    raw_sentences = re.split(r'(?<=[.!?;:])\s+', clean_text)
    
    safe_clauses = []
    for sent in raw_sentences:
        if len(sent) <= max_chars:
            safe_clauses.append(sent)
        else:
            sub_parts = re.split(r'(?<=[,-])\s+', sent)
            for part in sub_parts:
                if len(part) <= max_chars:
                    safe_clauses.append(part)
                else:
                    words = part.split(" ")
                    temp = ""
                    for w in words:
                        if len(temp) + len(w) + 1 <= max_chars:
                            temp = f"{temp} {w}".strip()
                        else:
                            safe_clauses.append(temp)
                            temp = w
                    if temp:
                        safe_clauses.append(temp)

    chunks = []
    current_chunk = ""
    for clause in safe_clauses:
        if len(current_chunk) + len(clause) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {clause}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = clause
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def get_audio_duration(wav_path: str) -> float:
    """Zwraca długość pliku WAV w sekundach."""
    with wave.open(str(wav_path), 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def measure_tts(name: str, generate_fn, output_path: str):
    """Mierzy czas wykonania oraz współczynnik RTF (Real-Time Factor) dla danego TTS."""
    print(f"\n>>> Testowanie: {name}")
    output_path_str = str(output_path)
    
    start_time = time.perf_counter()
    generate_fn(output_path_str)
    elapsed = time.perf_counter() - start_time

    duration = get_audio_duration(output_path_str)
    rtf = elapsed / duration if duration > 0 else 0
    print(f"[{name}] Wygenerowano w: {elapsed:.2f}s | Długość audio: {duration:.2f}s | RTF: {rtf:.3f}")

# ==========================================
# 1. PIPER TTS
# ==========================================
def download_piper_model(model_name: str, url_base: str) -> Path:
    for ext in [".onnx", ".onnx.json"]:
        filepath = BASE_DIR / (model_name + ext)
        if not filepath.exists():
            print(f"Pobieranie pliku modelu Piper: {filepath.name}...")
            urllib.request.urlretrieve(url_base + filepath.name, str(filepath))
    return BASE_DIR / f"{model_name}.onnx"

def run_piper(out_path: str):
    from piper.voice import PiperVoice
    
    model_name_en = "en_US-lessac-medium"
    url_base_en = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
    download_piper_model(model_name_en, url_base_en)

    model_name_pl = "pl_PL-gosia-medium"
    url_base_pl = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/"
    download_piper_model(model_name_pl, url_base_pl)
    
    voice = PiperVoice.load(f"{model_name_pl}.onnx")
    chunks = split_text_into_chunks(TEXT_PL, max_chars=200)
    print(f"[Piper] Podzielono tekst na {len(chunks)} fragmentów.")
    
    audio_segments = []
    sr = voice.config.sample_rate
    silence = np.zeros(int(sr * 0.15), dtype=np.float32)
    temp_chunk_path = OUTPUT_DIR / "temp_piper_chunk.wav"
    
    for i, chunk in enumerate(chunks, 1):
        print(f"  -> [Piper] Fragment {i}/{len(chunks)} ({len(chunk)} zn.): {chunk[:45]}...")
        with wave.open(str(temp_chunk_path), "wb") as wav_file:
            if hasattr(voice, "synthesize_wav"):
                voice.synthesize_wav(chunk, wav_file)
            else:
                voice.synthesize(chunk, wav_file)

        data, sr = sf_read_audio(str(temp_chunk_path))
        audio_segments.append(data)
        audio_segments.append(silence)

    if audio_segments:
        final_audio = np.concatenate(audio_segments)
        sf_write_audio(out_path, final_audio, sr)

    if temp_chunk_path.exists():
        temp_chunk_path.unlink()


# ==========================================
# 2. KOKORO TTS (z chunkingiem)
# ==========================================
def run_kokoro(out_path: str):
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code='a', repo_id="hexgrad/Kokoro-82M")

    chunks = split_text_into_chunks(TEXT_EN, max_chars=200)
    print(f"[Kokoro] Podzielono tekst na {len(chunks)} fragmentów.")
    audio_segments = []
    silence = np.zeros(int(24000 * 0.2), dtype=np.float32)

    for i, chunk in enumerate(chunks, 1):
        print(f"  -> [Kokoro] Fragment {i}/{len(chunks)} ({len(chunk)} zn.): {chunk[:45]}...")
        generator = pipeline(chunk, voice='af_heart', speed=1.0)
        for _, _, audio in generator:
            audio_segments.append(audio)
            audio_segments.append(silence)

    if audio_segments:
        final_audio = np.concatenate(audio_segments)
        sf_write_audio(out_path, final_audio, 24000)


# ==========================================
# 3. CHATTERBOX TTS (z chunkingiem)
# ==========================================
def run_chatterbox(out_path: str):
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelM = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    chunks = split_text_into_chunks(TEXT_PL, max_chars=180)
    print(f"[Chatterbox] Podzielono tekst na {len(chunks)} fragmentów.")
    audio_segments = []
    silence = np.zeros(int(modelM.sr * 0.2), dtype=np.float32)

    for i, chunk in enumerate(chunks, 1):
        print(f"  -> [Chatterbox] Fragment {i}/{len(chunks)} ({len(chunk)} zn.): {chunk[:45]}...")
        wav = modelM.generate(chunk, language_id="pl")
        audio_np = wav.squeeze().cpu().numpy()
        audio_segments.append(audio_np)
        audio_segments.append(silence)

    if audio_segments:
        final_audio = np.concatenate(audio_segments)
        sf_write_audio(out_path, final_audio, modelM.sr)


# ==========================================
# 4. COQUI XTTS-v2 (z chunkingiem i łatką torchaudio)
# ==========================================
def get_reference_speaker() -> str:
    """Zwraca ścieżkę do pliku referencyjnego dla klonowania głosu w XTTS-v2."""
    candidates = [
        BASE_DIR / "ref_voice.wav",
        OUTPUT_DIR / "kokoro.wav",
        OUTPUT_DIR / "piper.wav",
        OUTPUT_DIR / "chatterbox.wav",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError("Brak pliku referencyjnego .wav do klonowania głosu! Uruchom wcześniej np. Piper TTS.")


def run_xtts(out_path: str):
    import torch
    import torchaudio
    
    # Łatka naprawiająca błąd libtorchcodec / FFmpeg na Windowsie
    def _safe_load(filepath, *args, **kwargs):
        data, sr = sf_read_audio(filepath)
        tensor = torch.from_numpy(data)
        return (tensor.unsqueeze(0) if tensor.ndim == 1 else tensor.t()), sr

    def _safe_save(filepath, src, sample_rate, *args, **kwargs):
        sf_write_audio(filepath, src.detach().cpu().squeeze().numpy(), sample_rate)

    torchaudio.load = _safe_load
    torchaudio.save = _safe_save

    from TTS.api import TTS
    os.environ["COQUI_TOS_AGREED"] = "1"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speaker_sample = get_reference_speaker()
        
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    chunks = split_text_into_chunks(TEXT_PL, max_chars=180)
    print(f"[XTTS-v2] Podzielono tekst na {len(chunks)} fragmentów (próbka głosu: {Path(speaker_sample).name}).")
    audio_segments = []
    temp_chunk_path = OUTPUT_DIR / "temp_xtts_chunk.wav"
    
    for i, chunk in enumerate(chunks, 1):
        print(f"  -> [XTTS-v2] Fragment {i}/{len(chunks)} ({len(chunk)} zn.): {chunk[:45]}...")
        tts.tts_to_file(
            text=chunk,
            speaker_wav=speaker_sample,
            language="pl",
            file_path=str(temp_chunk_path),
            split_sentences=False
        )
        data, sr = sf_read_audio(str(temp_chunk_path))
        audio_segments.append(data)
        audio_segments.append(np.zeros(int(sr * 0.2), dtype=np.float32))
        
    if audio_segments:
        final_audio = np.concatenate(audio_segments)
        sf_write_audio(out_path, final_audio, sr)

    if temp_chunk_path.exists():
        temp_chunk_path.unlink()


# Pomocnicze bezpieczne funkcje zapisu/odczytu przez soundfile
def sf_write_audio(path, data, sr):
    import soundfile as sf
    sf.write(path, data, sr)


def sf_read_audio(path):
    import soundfile as sf
    return sf.read(str(path), dtype="float32")


if __name__ == "__main__":
    print("=== START TESTÓW TTS ===")

    TEXT_PL = load_text_safely(BASE_DIR / "cenie.txt", DEFAULT_TEXT_PL)
    TEXT_EN = load_text_safely(BASE_DIR / "hamlet.txt", DEFAULT_TEXT_EN)

    measure_tts("Piper TTS", run_piper, OUTPUT_DIR / "piper.wav")
    measure_tts("Kokoro TTS", run_kokoro, OUTPUT_DIR / "kokoro.wav")
    measure_tts("Chatterbox TTS", run_chatterbox, OUTPUT_DIR / "chatterbox.wav")
    measure_tts("Coqui XTTS-v2", run_xtts, OUTPUT_DIR / "xtts.wav")

    print("\n=== KONIEC TESTÓW ===")