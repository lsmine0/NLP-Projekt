from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
import urllib.request
import wave
from collections.abc import Callable, Iterable
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, ClassVar, Protocol, Self

DEFAULT_MODEL_NAME = "pl_PL-gosia-medium"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/"

_LOGGER = logging.getLogger(__name__)
_FINISHED = object()

_ABBREVIATIONS = (
    "np.",
    "itp.",
    "itd.",
    "tj.",
    "tzn.",
    "tzw.",
    "m.in.",
    "p.n.e.",
    "n.e.",
    "dr.",
    "prof.",
    "mgr.",
    "inż.",
    "hab.",
    "płk.",
    "gen.",
    "ul.",
    "al.",
    "nr.",
    "str.",
    "r.",
    "w.",
    "ww.",
    "tys.",
    "mln.",
    "mld.",
    "ok.",
    "por.",
    "zob.",
    "godz.",
    "min.",
    "sek.",
    "proc.",
    "św.",
    "ks.",
    "bp.",
    "o.",
    "oo.",
)

_THINK_START_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_END_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_MARKDOWN_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]+\)")
_URL_RE = re.compile(r"https?://\S+")
_TAG_RE = re.compile(r"<[^>]+>")
_MARKDOWN_RE = re.compile(r"[*_`#>~|]+")
_WHITESPACE_RE = re.compile(r"\s+")


class TTSError(RuntimeError):
    """Bazowy wyjątek błędów TTS."""


class TTSDependencyError(TTSError):
    """Wymagana biblioteka TTS nie jest zainstalowana."""


class TTSModelError(TTSError):
    """Model Piper jest niedostępny lub nie można go załadować."""


class AudioSink(Protocol):
    """Odbiorca strumienia PCM; GUI może podać własny sink, np. WebSocket."""

    def open(self, sample_rate: int, channels: int, sample_width: int) -> None: ...

    def write(self, pcm: bytes) -> None: ...

    def close(self) -> None: ...

    def abort(self) -> None: ...


class SoundDeviceSink:
    """Odtwarzanie PCM przez sounddevice."""

    _DTYPES: ClassVar[dict[int, str]] = {1: "int8", 2: "int16", 4: "int32"}

    def __init__(self, device: int | str | None = None) -> None:
        self.device = device
        self._stream: Any | None = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._buffer = bytearray()
        self._sample_rate = 0
        self._channels = 1
        self._sample_width = 2
        self._max_buffer = 0
        self._aborted = False
        self._callback_abort: type[Exception] | None = None

    def open(self, sample_rate: int, channels: int, sample_width: int) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise TTSDependencyError(
                "Brak pakietu sounddevice. Zainstaluj: pip install sounddevice."
            ) from exc

        dtype = self._DTYPES.get(sample_width)
        if dtype is None:
            raise TTSError(f"Nieobsługiwana szerokość próbki: {sample_width} B.")

        with self._condition:
            if self._stream is not None:
                return
            self._sample_rate = sample_rate
            self._channels = channels
            self._sample_width = sample_width
            self._max_buffer = max(1, sample_rate * channels * sample_width // 2)
            self._buffer.clear()
            self._aborted = False
            self._callback_abort = sd.CallbackAbort

        stream = sd.RawOutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype,
            device=self.device,
            callback=self._audio_callback,
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise

        with self._condition:
            self._stream = stream
            aborted = self._aborted
        if aborted:
            stream.abort()
            stream.close()
            raise TTSError("Odtwarzanie zostało przerwane.")

    def _audio_callback(
        self, outdata: Any, frames: int, _time: Any, _status: Any
    ) -> None:
        needed = frames * self._channels * self._sample_width
        with self._condition:
            if self._aborted:
                outdata[:] = b"\x00" * needed
                if self._callback_abort is not None:
                    raise self._callback_abort
            if len(self._buffer) >= needed:
                data = bytes(self._buffer[:needed])
                del self._buffer[:needed]
            else:
                data = bytes(self._buffer) + b"\x00" * (needed - len(self._buffer))
                self._buffer.clear()
            self._condition.notify_all()
        outdata[:] = data

    def write(self, pcm: bytes) -> None:
        view = memoryview(pcm)
        while view:
            with self._condition:
                while not self._aborted and len(self._buffer) >= self._max_buffer:
                    self._condition.wait(0.1)
                if self._aborted:
                    raise TTSError("Odtwarzanie zostało przerwane.")
                space = self._max_buffer - len(self._buffer)
                if space <= 0:
                    continue
                count = min(space, len(view))
                self._buffer.extend(view[:count])
                view = view[count:]
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            while not self._aborted and self._buffer:
                self._condition.wait(0.1)
            stream = self._stream
            self._stream = None
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()

    def abort(self) -> None:
        with self._condition:
            self._aborted = True
            self._buffer.clear()
            self._condition.notify_all()
            stream = self._stream
            self._stream = None
        if stream is None:
            return
        try:
            stream.abort()
        finally:
            stream.close()


class WavFileSink:
    """Zapis PCM do pliku WAV, używany przez skrypt testowy."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._wave: wave.Wave_write | None = None
        self._lock = threading.Lock()

    def open(self, sample_rate: int, channels: int, sample_width: int) -> None:
        with self._lock:
            if self._wave is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            wav_file = wave.open(str(self.path), "wb")
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            self._wave = wav_file

    def write(self, pcm: bytes) -> None:
        with self._lock:
            if self._wave is None:
                raise TTSError("Plik WAV nie jest otwarty.")
            self._wave.writeframes(pcm)

    def close(self) -> None:
        with self._lock:
            wav_file = self._wave
            self._wave = None
        if wav_file is not None:
            wav_file.close()

    def abort(self) -> None:
        self.close()


def _sanitize_for_speech(text: str) -> str:
    text = _CODE_BLOCK_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _MARKDOWN_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _ends_with_abbreviation(prefix: str) -> bool:
    candidate = f"{prefix}.".lower()
    if candidate.endswith(_ABBREVIATIONS):
        return True
    return bool(re.search(r"(?:^|\s)[a-z]\.$", candidate))


class SentenceSplitter:
    """Strumieniowy podział tekstu na fragmenty odpowiednie dla TTS."""

    def __init__(self, max_chunk_chars: int = 220, min_chunk_chars: int = 40) -> None:
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars musi być większe od zera.")
        if min_chunk_chars < 0 or min_chunk_chars >= max_chunk_chars:
            raise ValueError("min_chunk_chars musi być z zakresu 0..max_chunk_chars.")
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self._buffer = ""
        self._in_think = False
        self._tag_tail = ""

    def feed(self, fragment: str) -> list[str]:
        text = self._strip_think(fragment)
        if not text:
            return []
        self._buffer += text
        return self._drain()

    def flush(self) -> list[str]:
        self._in_think = False
        self._tag_tail = ""
        text = _sanitize_for_speech(self._buffer)
        self._buffer = ""
        return [text] if text else []

    def _strip_think(self, fragment: str) -> str:
        text = self._tag_tail + fragment
        self._tag_tail = ""
        output: list[str] = []
        while text:
            if self._in_think:
                match = _THINK_END_RE.search(text)
                if match is not None:
                    text = text[match.end() :]
                    self._in_think = False
                    continue
                tail = self._partial_tag_start(text, closing=True)
                if tail is not None:
                    self._tag_tail = text[tail:]
                return "".join(output)

            match = _THINK_START_RE.search(text)
            if match is not None:
                output.append(text[: match.start()])
                text = text[match.end() :]
                self._in_think = True
                continue

            tail = self._partial_tag_start(text, closing=False)
            if tail is None:
                output.append(text)
            else:
                output.append(text[:tail])
                self._tag_tail = text[tail:]
            return "".join(output)
        return "".join(output)

    @staticmethod
    def _partial_tag_start(text: str, closing: bool) -> int | None:
        name = "</think" if closing else "<think"
        index = text.rfind("<")
        while index != -1:
            suffix = text[index:].lower()
            if name.startswith(suffix):
                return index
            if suffix.startswith(name):
                rest = suffix[len(name) :]
                if not rest or rest[0] in " \t\r\n>/":
                    return index
            index = text.rfind("<", 0, index)
        return None

    def _drain(self) -> list[str]:
        chunks: list[str] = []
        while True:
            boundary = self._find_sentence_boundary()
            if boundary is not None:
                chunk = self._buffer[:boundary]
                self._buffer = self._buffer[boundary:].lstrip()
                cleaned = _sanitize_for_speech(chunk)
                if cleaned:
                    chunks.append(cleaned)
                continue
            if len(self._buffer) <= self.max_chunk_chars:
                break
            cut = self._find_forced_cut()
            chunk = self._buffer[:cut]
            self._buffer = self._buffer[cut:].lstrip()
            cleaned = _sanitize_for_speech(chunk)
            if cleaned:
                chunks.append(cleaned)
        return chunks

    def _find_sentence_boundary(self) -> int | None:
        buffer = self._buffer
        index = 0
        while index < len(buffer):
            char = buffer[index]
            if char == "\n":
                return index + 1
            if char in ".!?…":
                end = index + 1
                while end < len(buffer) and buffer[end] in ".!?…":
                    end += 1
                after = end
                while after < len(buffer) and buffer[after] in "\"'”’)]»":
                    after += 1
                if after >= len(buffer):
                    return None
                if buffer[after].isspace() and self._is_safe_boundary(
                    buffer[:index], buffer[index:end]
                ):
                    return after
                index = after
                continue
            index += 1
        return None

    def _is_safe_boundary(self, prefix: str, terminator: str) -> bool:
        if not prefix.strip():
            return False
        return not (
            terminator == "." and _ends_with_abbreviation(prefix.rstrip())
        )

    def _find_forced_cut(self) -> int:
        limit = min(self.max_chunk_chars, len(self._buffer))
        cut = -1
        for separator in ("\n", " ", ",", ";", ":"):
            cut = max(cut, self._buffer.rfind(separator, 0, limit))
        if cut < self.min_chunk_chars:
            return max(1, limit)
        if self._buffer[cut] in ",;:":
            return cut + 1
        return cut


class TTSHandle:
    """Generuje i odtwarza mowę, dzieląc tekst na zdania."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_dir: str | Path | None = None,
        auto_download: bool = True,
        sink: AudioSink | None = None,
        max_chunk_chars: int = 220,
        length_scale: float | None = None,
        volume: float = 1.0,
        stop_timeout: float = 1.5,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars musi być większe od zera.")
        if length_scale is not None and length_scale <= 0:
            raise ValueError("length_scale musi być większe od zera.")
        if volume <= 0:
            raise ValueError("volume musi być większe od zera.")
        if stop_timeout <= 0:
            raise ValueError("stop_timeout musi być większe od zera.")

        self.model_dir = (
            Path(model_dir).expanduser() if model_dir is not None else DEFAULT_MODEL_DIR
        )
        self.model_path = self._resolve_model_path(model_path, self.model_dir)
        self.config_path = Path(f"{self.model_path}.json")
        self.auto_download = auto_download
        self.max_chunk_chars = max_chunk_chars
        self.length_scale = length_scale
        self.volume = volume
        self.stop_timeout = stop_timeout
        self._sink = sink if sink is not None else SoundDeviceSink()
        self._on_error = on_error

        self._voice: Any | None = None
        self._synthesis_config: Any | None = None
        self._voice_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()

        self._generation = 0
        self._cancel_event: threading.Event | None = None
        self._threads: list[threading.Thread] = []
        self._sentence_queue: Queue[Any] | None = None
        self._audio_queue: Queue[Any] | None = None
        self._is_speaking = False
        self._last_error: Exception | None = None

    @staticmethod
    def _resolve_model_path(
        model_path: str | Path | None, model_dir: Path
    ) -> Path:
        if model_path is None:
            return model_dir / f"{DEFAULT_MODEL_NAME}.onnx"

        path = Path(model_path).expanduser()
        if path.suffix != ".onnx":
            path = (
                path / f"{DEFAULT_MODEL_NAME}.onnx"
                if path.is_dir()
                else path.with_name(f"{path.name}.onnx")
            )
        if not path.is_absolute():
            path = path.resolve() if path.exists() else model_dir / path.name
        return path

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(self._get_voice().session.get_providers())

    @property
    def sample_rate(self) -> int:
        return int(self._get_voice().config.sample_rate)

    def warmup(self) -> None:
        """Ładuje model i wykonuje krótką syntezę bez odtwarzania."""
        # GUI/orchestrator: wywołać przy starcie, aby usunąć opóźnienie pierwszego zdania.
        voice = self._get_voice()
        with self._model_lock:
            next(iter(voice.synthesize("Dzień dobry", syn_config=self._synthesis_config)), None)

    def speak(self, text: str) -> None:
        self.speak_stream((text,))

    def speak_stream(self, tokens: Iterable[str]) -> None:
        """Czyta iterator tokenów; GUI/orchestrator może przekazać strumień LLM."""
        if isinstance(tokens, str):
            tokens = (tokens,)

        with self._lifecycle_lock:
            self._cancel_current()
            self._generation += 1
            generation = self._generation
            cancel = threading.Event()
            self._cancel_event = cancel
            self._last_error = None
            self._is_speaking = True

            sentence_queue: Queue[Any] = Queue(maxsize=8)
            audio_queue: Queue[Any] = Queue(maxsize=8)
            self._sentence_queue = sentence_queue
            self._audio_queue = audio_queue
            synth_done = threading.Event()
            player_done = threading.Event()

            threads = [
                threading.Thread(
                    target=self._read_worker,
                    args=(generation, cancel, tokens, sentence_queue, synth_done),
                    name="tts-reader",
                    daemon=True,
                ),
                threading.Thread(
                    target=self._synth_worker,
                    args=(
                        generation,
                        cancel,
                        sentence_queue,
                        audio_queue,
                        player_done,
                        synth_done,
                    ),
                    name="tts-synth",
                    daemon=True,
                ),
                threading.Thread(
                    target=self._play_worker,
                    args=(generation, cancel, audio_queue, player_done),
                    name="tts-player",
                    daemon=True,
                ),
            ]
            self._threads = threads
            for thread in threads:
                thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._cancel_current()

    def wait(self, timeout: float | None = None) -> bool:
        threads = list(self._threads)
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in threads:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(remaining)
        return not any(thread.is_alive() for thread in threads)

    def close(self) -> None:
        self.stop()
        self._abort_sink()
        with self._voice_lock:
            self._voice = None
            self._synthesis_config = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _cancel_current(self) -> None:
        self._is_speaking = False
        cancel = self._cancel_event
        if cancel is not None:
            cancel.set()
        self._cancel_event = None
        self._abort_sink()
        self._drain_queue(self._sentence_queue)
        self._drain_queue(self._audio_queue)

        threads = self._threads
        self._threads = []
        deadline = time.monotonic() + self.stop_timeout
        for thread in threads:
            if thread is threading.current_thread():
                continue
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(remaining)

    def _abort_sink(self) -> None:
        try:
            self._sink.abort()
        except Exception as exc:
            self._record_error(exc)

    @staticmethod
    def _drain_queue(queue: Queue[Any] | None) -> None:
        if queue is None:
            return
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return

    def _read_worker(
        self,
        generation: int,
        cancel: threading.Event,
        tokens: Iterable[str],
        sentence_queue: Queue[Any],
        synth_done: threading.Event,
    ) -> None:
        splitter = SentenceSplitter(self.max_chunk_chars)
        try:
            for token in tokens:
                if cancel.is_set() or generation != self._generation:
                    break
                for sentence in splitter.feed(str(token)):
                    if not self._put(sentence_queue, sentence, cancel, generation):
                        return
            if not cancel.is_set() and generation == self._generation:
                for sentence in splitter.flush():
                    if not self._put(sentence_queue, sentence, cancel, generation):
                        return
        except Exception as exc:
            if not cancel.is_set() and generation == self._generation:
                self._record_error(exc)
        finally:
            self._put_final(sentence_queue, cancel, generation, synth_done)

    def _synth_worker(
        self,
        generation: int,
        cancel: threading.Event,
        sentence_queue: Queue[Any],
        audio_queue: Queue[Any],
        player_done: threading.Event,
        synth_done: threading.Event,
    ) -> None:
        try:
            while not cancel.is_set() and generation == self._generation:
                sentence = self._get(sentence_queue, cancel)
                if sentence is _FINISHED:
                    break
                voice = self._get_voice()
                with self._model_lock:
                    if cancel.is_set() or generation != self._generation:
                        break
                    for chunk in voice.synthesize(
                        sentence, syn_config=self._synthesis_config
                    ):
                        if cancel.is_set() or generation != self._generation:
                            break
                        pcm = chunk.audio_int16_bytes
                        if not pcm:
                            continue
                        item = (
                            int(chunk.sample_rate),
                            int(chunk.sample_channels),
                            int(chunk.sample_width),
                            pcm,
                        )
                        if not self._put(audio_queue, item, cancel, generation):
                            return
        except Exception as exc:
            if not cancel.is_set() and generation == self._generation:
                self._record_error(exc)
        finally:
            try:
                self._put_final(audio_queue, cancel, generation, player_done)
            finally:
                synth_done.set()

    def _play_worker(
        self,
        generation: int,
        cancel: threading.Event,
        audio_queue: Queue[Any],
        player_done: threading.Event,
    ) -> None:
        opened = False
        try:
            while not cancel.is_set() and generation == self._generation:
                item = self._get(audio_queue, cancel)
                if item is _FINISHED:
                    break
                sample_rate, channels, sample_width, pcm = item
                if not opened:
                    self._sink.open(sample_rate, channels, sample_width)
                    opened = True
                if cancel.is_set() or generation != self._generation:
                    break
                self._sink.write(pcm)
        except Exception as exc:
            if not cancel.is_set() and generation == self._generation:
                self._record_error(exc)
        finally:
            if opened and generation == self._generation:
                try:
                    self._sink.close()
                except Exception as exc:
                    if not cancel.is_set() and generation == self._generation:
                        self._record_error(exc)
            if generation == self._generation:
                self._is_speaking = False
            player_done.set()

    def _put(
        self,
        queue: Queue[Any],
        item: Any,
        cancel: threading.Event,
        generation: int,
    ) -> bool:
        while not cancel.is_set() and generation == self._generation:
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def _put_final(
        self,
        queue: Queue[Any],
        cancel: threading.Event,
        generation: int,
        consumer_done: threading.Event,
    ) -> None:
        while (
            not cancel.is_set()
            and generation == self._generation
            and not consumer_done.is_set()
        ):
            try:
                queue.put(_FINISHED, timeout=0.1)
                return
            except Full:
                continue
        try:
            queue.put_nowait(_FINISHED)
        except Full:
            pass

    @staticmethod
    def _get(queue: Queue[Any], cancel: threading.Event) -> Any:
        while not cancel.is_set():
            try:
                return queue.get(timeout=0.1)
            except Empty:
                continue
        return _FINISHED

    def _record_error(self, exc: Exception) -> None:
        self._last_error = exc
        if self._on_error is not None:
            try:
                self._on_error(exc)
            except Exception:
                pass

    def _ensure_model(self) -> None:
        if self.model_path.exists() and self.config_path.exists():
            return
        if not self.auto_download:
            raise TTSModelError(
                f"Brak modelu Piper: {self.model_path}. "
                "Włącz auto_download lub wskaż istniejący plik."
            )
        if self.model_path.name != f"{DEFAULT_MODEL_NAME}.onnx":
            raise TTSModelError(
                f"Brak modelu Piper: {self.model_path}. "
                "Automatyczne pobieranie obsługuje tylko domyślny model."
            )

        if not self.model_path.exists():
            self._download_model(self.model_path.name, self.model_path)
        if not self.config_path.exists():
            self._download_model(f"{self.model_path.name}.json", self.config_path)

    @staticmethod
    def _download_model(file_name: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.part")
        url = f"{MODEL_URL}{file_name}"
        _LOGGER.info("Pobieranie modelu Piper: %s", url)
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("wb") as output,
            ):
                shutil.copyfileobj(response, output, length=1024 * 1024)
            os.replace(temporary, target)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise TTSModelError(
                f"Nie udało się pobrać modelu {file_name}: {exc}"
            ) from exc

    def _get_voice(self) -> Any:
        with self._voice_lock:
            if self._voice is not None:
                return self._voice

            self._ensure_model()
            try:
                from piper import PiperVoice, SynthesisConfig
            except ImportError as exc:
                raise TTSDependencyError(
                    "Brak pakietu piper-tts. Zainstaluj: pip install piper-tts."
                ) from exc

            try:
                voice = PiperVoice.load(str(self.model_path), use_cuda=False)
            except Exception as exc:
                raise TTSModelError(
                    f"Nie można załadować modelu Piper {self.model_path}: {exc}"
                ) from exc

            self._voice = voice
            self._synthesis_config = SynthesisConfig(
                length_scale=self.length_scale,
                volume=self.volume,
            )
            return self._voice
