"""Mikrofonowe nagrywanie i transkrypcja mowy z użyciem Faster-Whisper."""

import ctypes
import os
import site
from pathlib import Path
from typing import Any


class STTError(RuntimeError):
    """Bazowy wyjątek błędów obsługi STT."""


class STTDependencyError(STTError):
    """Wymagana biblioteka STT nie jest zainstalowana."""


class NoInputDeviceError(STTError):
    """Nie znaleziono lub nie można otworzyć wejścia audio."""


class EmptyRecordingError(STTError):
    """Nagranie nie zawiera żadnych próbek audio."""


class SilenceDetectedError(STTError):
    """Nagranie jest ciche albo nie udało się rozpoznać w nim mowy."""


class RecordingStateError(STTError):
    """Operację nagrywania wywołano w nieprawidłowym stanie."""


_CUDA_DLL_HANDLES: list[Any] = []


def _configure_cuda_dll_paths() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    for package_dir in site.getsitepackages():
        for library in ("cublas", "cudnn"):
            dll_path = Path(package_dir) / "nvidia" / library / "bin"
            if dll_path.is_dir():
                path = str(dll_path)
                _CUDA_DLL_HANDLES.append(os.add_dll_directory(path))
                os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


def _cuda_runtime_available() -> bool:
    if os.name != "nt":
        return True

    try:
        _CUDA_DLL_HANDLES.extend(
            ctypes.WinDLL(name) for name in ("cublas64_12.dll", "cudnn64_9.dll")
        )
    except OSError:
        return False
    return True


class STTHandle:
    """Nagrywa dźwięk z mikrofonu i zwraca transkrypcję jako tekst.

    Nagrywanie jest asynchroniczne: wywołaj ``start_recording`` przy rozpoczęciu
    akcji w GUI, a następnie ``stop_recording`` i odbierz gotowy tekst.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str | None = None,
        compute_type: str | None = None,
        sample_rate: int = 16_000,
        input_device: int | str | None = None,
        language: str | None = "pl",
        silence_threshold: float = 0.001,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate musi być większe od zera.")
        if not 0 <= silence_threshold <= 1:
            raise ValueError("silence_threshold musi być z zakresu 0..1.")

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.language = language
        self.silence_threshold = silence_threshold
        self._stream: Any | None = None
        self._audio_chunks: list[Any] = []
        self._recording_status: str | None = None
        self._model: Any | None = None

    @property
    def is_recording(self) -> bool:
        """Informuje, czy trwa nagrywanie."""
        return self._stream is not None

    @staticmethod
    def _sounddevice() -> Any:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise STTDependencyError(
                "Brak pakietu sounddevice. Zainstaluj zależności z tests/requirements-stt.txt."
            ) from exc
        return sd

    def start_recording(self) -> None:
        """Rozpoczyna nagrywanie i natychmiast wraca do wywołującego."""
        if self.is_recording:
            raise RecordingStateError("Nagrywanie już trwa.")

        sd = self._sounddevice()
        try:
            device_info = sd.query_devices(device=self.input_device, kind="input")
        except sd.PortAudioError as exc:
            raise NoInputDeviceError("Nie znaleziono urządzenia wejściowego.") from exc

        if device_info["max_input_channels"] < 1:
            raise NoInputDeviceError("Wybrane urządzenie nie obsługuje wejścia audio.")

        self._audio_chunks = []
        self._recording_status = None
        stream = sd.InputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        try:
            stream.start()
        except sd.PortAudioError as exc:
            stream.close()
            raise NoInputDeviceError("Nie można rozpocząć nagrywania z mikrofonu.") from exc
        self._stream = stream

    def _audio_callback(self, audio: Any, *callback_args: Any) -> None:
        status = callback_args[-1] if callback_args else None
        if status:
            self._recording_status = str(status)
        self._audio_chunks.append(audio.copy())

    def stop_recording(self) -> str:
        """Kończy nagrywanie i zwraca transkrypcję jako ciąg znaków."""
        if not self.is_recording:
            raise RecordingStateError("Nagrywanie nie zostało rozpoczęte.")

        stream = self._stream
        self._stream = None
        sd = self._sounddevice()
        try:
            stream.stop()
        except sd.PortAudioError as exc:
            raise STTError("Nie można poprawnie zakończyć nagrywania audio.") from exc
        finally:
            stream.close()

        if self._recording_status is not None:
            raise STTError(f"Błąd podczas nagrywania audio: {self._recording_status}")
        if not self._audio_chunks:
            raise EmptyRecordingError("Nagranie nie zawiera próbek audio.")

        try:
            import numpy as np
        except ImportError as exc:
            raise STTDependencyError(
                "Brak pakietu numpy. Zainstaluj zależności z tests/requirements-stt.txt."
            ) from exc

        audio = np.concatenate(self._audio_chunks, axis=0).reshape(-1)
        if audio.size == 0:
            raise EmptyRecordingError("Nagranie nie zawiera próbek audio.")
        if float(np.max(np.abs(audio))) <= self.silence_threshold:
            raise SilenceDetectedError("W nagraniu wykryto ciszę.")

        segments, _ = self._get_model().transcribe(
            audio,
            beam_size=5,
            language=self.language,
        )
        transcription = " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        ).strip()
        if not transcription:
            raise SilenceDetectedError("Nie rozpoznano mowy w nagraniu.")
        return transcription

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        _configure_cuda_dll_paths()
        try:
            import ctranslate2
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise STTDependencyError(
                "Brak pakietu faster-whisper. "
                "Zainstaluj zależności z tests/requirements-stt.txt."
            ) from exc

        device = self.device
        if device is None:
            has_cuda_device = ctranslate2.get_cuda_device_count() > 0
            has_cuda_runtime = has_cuda_device and _cuda_runtime_available()
            device = "cuda" if has_cuda_runtime else "cpu"
        elif device == "cuda" and not _cuda_runtime_available():
            raise STTDependencyError(
                "Brak wymaganych bibliotek CUDA (cuBLAS 12 i cuDNN 9). "
                "Zainstaluj je lub ustaw device='cpu'."
            )
        compute_type = self.compute_type
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        self._model = WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type,
        )
        return self._model