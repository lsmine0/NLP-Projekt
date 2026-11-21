"""Runtime-only configuration. Credentials never pass through SettingsStore."""
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class RuntimeConfig:
    endpoint: str = 'http://127.0.0.1:11434/v1'
    model: str = ''
    api_key: str = field(default='ollam', repr=False)
    request_timeout: float = 60.0
    workspace: Path = field(default_factory=Path.cwd)
    data_dir: Path = field(default_factory=lambda: Path('data'))
    provider_name: str = 'ollama'

    def __post_init__(self):
        parsed = urlsplit(self.endpoint)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('Endpoint musi być adresem HTTP(S) bez danych logowania, query i fragmentu.')
        if not math.isfinite(self.request_timeout) or self.request_timeout <= 0:
            raise ValueError('Timeout musi być dodatnią, skończoną liczbą.')
        if self.provider_name not in ('ollama', 'local', 'openai'):
            raise ValueError('Nieobsługiwany provider.')
        if self.provider_name == 'openai' and self.api_key == 'ollam':
            raise ValueError('Brak klucza API (OPENAI_API_KEY).')
        object.__setattr__(self, 'workspace', Path(self.workspace).expanduser().resolve())
        object.__setattr__(self, 'data_dir', Path(self.data_dir).expanduser().resolve())

    @classmethod
    def from_env(cls, settings=None, *, provider_name=None, environ=None):
        env = os.environ if environ is None else environ
        cfg = settings or {}
        return cls(
            endpoint=env.get('NLP_ENDPOINT') or cfg.get('ollama_endpoint') or env.get('OLLAMA_ENDPOINT') or cls.endpoint,
            model=env.get('NLP_MODEL') or cfg.get('ollama_model') or env.get('OLLAMA_MODEL', ''),
            api_key=env.get('OPENAI_API_KEY') or env.get('OLLAMA_API_KEY') or 'ollam',
            request_timeout=float(env.get('NLP_REQUEST_TIMEOUT', '60')),
            workspace=Path(env.get('NLP_WORKSPACE') or Path.cwd()),
            data_dir=Path(env.get('CHAT_DATA_DIR') or Path(__file__).resolve().parents[1] / 'data'),
            provider_name=provider_name or env.get('NLP_PROVIDER', 'ollama'),
        )
