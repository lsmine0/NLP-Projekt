import argparse
import logging
import re
import sys
import time
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handlers.TTSHandle import (
    AudioSink,
    SoundDeviceSink,
    TTSError,
    TTSHandle,
    WavFileSink,
)

DEFAULT_TEXT = (
    "Mój stary to fanatyk wędkarstwa. Pół mieszkania zawalone wędkami, najgorzej. Średnio raz w miesiącu ktoś wdepnie w leżący na ziemi haczyk czy kotwicę i trzeba wyciągać w szpitalu, bo mają zadziory. "
    "W swoim 22-letnim życiu już 10 razy byłem na takim zabiegu. "
    "Tydzień temu poszedłem na jakieś losowe badania, to baba z recepcji jak mnie tylko zobaczyła, to kazała buta ściągać, bo myślała, że znowu hak w nodze."
)


class TimingSink:
    def __init__(self, inner: AudioSink) -> None:
        self.inner = inner
        self.started_at: float | None = None
        self.first_audio_at: float | None = None
        self.sample_rate: int | None = None
        self.channels: int | None = None
        self.sample_width: int | None = None
        self.bytes_written = 0

    def open(self, sample_rate: int, channels: int, sample_width: int) -> None:
        self.started_at = time.perf_counter()
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.inner.open(sample_rate, channels, sample_width)

    def write(self, pcm: bytes) -> None:
        if self.first_audio_at is None:
            self.first_audio_at = time.perf_counter()
        self.bytes_written += len(pcm)
        self.inner.write(pcm)

    def close(self) -> None:
        self.inner.close()

    def abort(self) -> None:
        self.inner.abort()

    @property
    def audio_seconds(self) -> float:
        if not self.sample_rate or not self.channels or not self.sample_width:
            return 0.0
        return self.bytes_written / (
            self.sample_rate * self.channels * self.sample_width
        )


def token_stream(text: str, delay: float) -> Iterable[str]:
    for token in re.findall(r"\S+\s*", text):
        yield token
        if delay > 0:
            time.sleep(delay)


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1250", "iso-8859-2"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Nie udało się rozpoznać kodowania pliku: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ręczny test TTSHandle.")
    parser.add_argument("--model", help="Ścieżka do modelu .onnx.")
    parser.add_argument("--text", help="Tekst do wypowiedzenia.")
    parser.add_argument("--file", type=Path, help="Plik tekstowy z wypowiedzią.")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Symuluj strumień tokenów zamiast przekazywać cały tekst.",
    )
    parser.add_argument(
        "--token-delay",
        type=float,
        default=0.02,
        help="Opóźnienie między tokenami w trybie --stream (sekundy).",
    )
    parser.add_argument(
        "--stop-after",
        type=float,
        help="Przerwij odtwarzanie po podanej liczbie sekund.",
    )
    parser.add_argument("--wav", type=Path, help="Zapisz audio do pliku WAV.")
    return parser.parse_args()


def print_metrics(
    label: str,
    started_at: float,
    sink: TimingSink,
    elapsed: float,
) -> None:
    first_audio = (
        sink.first_audio_at - started_at if sink.first_audio_at is not None else None
    )
    duration = sink.audio_seconds
    print(f"\n[{label}]")
    if first_audio is None:
        print("  Brak audio.")
    else:
        print(f"  Czas do pierwszego audio: {first_audio:.3f} s")
    print(f"  Łączny czas: {elapsed:.3f} s")
    print(f"  Długość audio: {duration:.3f} s")
    if duration > 0:
        print(f"  RTF: {elapsed / duration:.3f}")
    if sink.sample_rate:
        print(
            f"  Format: {sink.sample_rate} Hz, {sink.channels} kanał(y), "
            f"{sink.sample_width * 8} bit"
        )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.file is not None:
        text = read_text(args.file)
    elif args.text is not None:
        text = args.text
    else:
        text = DEFAULT_TEXT

    if not text.strip():
        print("Tekst jest pusty.")
        return 2

    inner_sink: AudioSink = (
        WavFileSink(args.wav) if args.wav is not None else SoundDeviceSink()
    )
    sink = TimingSink(inner_sink)
    handle = TTSHandle(
        model_path=args.model,
        sink=sink,
        on_error=lambda exc: print(f"Błąd TTS w wątku: {exc}"),
    )

    try:
        print(f"Model: {handle.model_path}")
        print("Ładowanie modelu i rozgrzewka...")
        warmup_started = time.perf_counter()
        handle.warmup()
        print(f"Model gotowy w {time.perf_counter() - warmup_started:.3f} s.")
        print(f"Provider ONNX: {', '.join(handle.providers)}")
        print(f"Sample rate modelu: {handle.sample_rate} Hz")
        if handle.providers != ("CPUExecutionProvider",):
            print("Błąd: model nie działa wyłącznie na CPU.")
            return 3

        started_at = time.perf_counter()
        if args.stream:
            handle.speak_stream(token_stream(text, args.token_delay))
        else:
            handle.speak(text)

        if args.stop_after is not None:
            time.sleep(args.stop_after)
            stop_started = time.perf_counter()
            handle.stop()
            print(f"\nstop() zakończone w {time.perf_counter() - stop_started:.3f} s.")
            print(f"is_speaking po stop(): {handle.is_speaking}")
            print_metrics("PRZERWANE", started_at, sink, time.perf_counter() - started_at)

            sink.bytes_written = 0
            sink.first_audio_at = None
            sink.started_at = None
            started_at = time.perf_counter()
            handle.speak("Przerwano odtwarzanie. To jest test ponownego uruchomienia.")
            handle.wait()
            print_metrics(
                "PO STOP", started_at, sink, time.perf_counter() - started_at
            )
        else:
            completed = handle.wait()
            print_metrics(
                "TTS", started_at, sink, time.perf_counter() - started_at
            )
            print(f"  wait() zakończone: {completed}")

        if handle.last_error is not None:
            print(f"Ostatni błąd: {handle.last_error}")
            return 1
        if args.wav is not None:
            print(f"Zapisano WAV: {args.wav}")
        return 0
    except (TTSError, ValueError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print("\nPrzerwano przez użytkownika.")
        else:
            print(f"Błąd TTS: {exc}")
        return 1
    finally:
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
