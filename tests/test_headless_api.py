"""Offline tests exercising the real ASGI lifespan and real AgentFlow with a fake provider."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.api import create_app
from core.main import build_api_runtime
from core.RuntimeConfig import RuntimeConfig
from handlers.ToolHandle import ToolHandle


class FakeProvider:
    def __init__(self, config, tools):
        self.config = config
        self.ToolsHandle = tools
        self.base_url = config.endpoint
        self.model = None
        self.closed = 0
        self.available = True
        self.calls = 0
        self.chats = []

    def GetModels(self):
        self.calls += 1
        if not self.available:
            raise RuntimeError('private-secret http://private-host')
        return [NS(id='test-model')]

    def SetModel(self, name):
        self.model = name
        return True

    def GetNumCtx(self):
        return 16384

    def EnsureContextBudget(self, messages, **kwargs):
        return None

    def ChatStream(self, messages, **kwargs):
        self.chats.append(messages)
        yield NS(choices=[NS(delta=NS(content='Odpowiedź testowa', tool_calls=None,
                                    reasoning_content=None))])

    def close(self):
        self.closed += 1


def setup_app(tmp_path, **config_changes):
    config = RuntimeConfig(data_dir=tmp_path / 'data', workspace=tmp_path,
                           model='test-model', api_key='private-secret', **config_changes)
    made = []
    def make(config, tools):
        provider = FakeProvider(config, tools)
        made.append(provider)
        return provider
    def factory():
        return build_api_runtime(config, provider_factory=make, tool_factory=ToolHandle)
    return create_app(runtime_factory=factory), made


def test_lifespan_injection_stream_and_shutdown(tmp_path):
    app, providers = setup_app(tmp_path)
    assert providers == []  # no construction or network on import/factory creation
    with TestClient(app) as client:
        assert len(providers) == 1
        assert providers[0].calls == 0
        assert client.get('/health/live').json() == {'status': 'alive'}
        assert providers[0].calls == 0
        assert client.get('/health/ready').status_code == 200
        response = client.post('/api/chat', json={'message': 'Cześć'})
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert {'type': 'token', 'text': 'Odpowiedź testowa'} in events
        assert len(providers) == 2
        agent = next(iter(app.state.api.sessions.values()))
        assert agent.flow.ollama is providers[1]
        assert providers[1].chats
    assert all(p.closed == 1 for p in providers)
    assert not hasattr(app.state, 'api')


def test_liveness_does_not_depend_on_model_and_readiness_is_redacted(tmp_path):
    app, providers = setup_app(tmp_path)
    with TestClient(app) as client:
        providers[0].available = False
        assert client.get('/health/live').status_code == 200
        response = client.get('/health/ready')
        assert response.status_code == 503
        assert response.json()['components']['provider'] is False
        assert 'private-secret' not in response.text
        assert 'private-host' not in response.text
        assert client.get('/api/models').status_code == 502
        assert 'private-secret' not in client.get('/api/models').text
        providers[0].available = True
        assert client.get('/health/ready').status_code == 200


def test_missing_configured_model_is_not_ready(tmp_path):
    app, providers = setup_app(tmp_path)
    with TestClient(app) as client:
        providers[0].GetModels = lambda: [NS(id='different')]
        assert client.get('/health/ready').status_code == 503


def test_two_apps_do_not_share_sessions_or_storage(tmp_path):
    left, lp = setup_app(tmp_path / 'left')
    right, rp = setup_app(tmp_path / 'right')
    with TestClient(left) as one, TestClient(right) as two:
        one.post('/api/chat', json={'message': 'Pierwsza'})
        assert len(one.get('/api/chats').json()) == 1
        assert two.get('/api/chats').json() == []
        assert left.state.api.sessions is not right.state.api.sessions
        two.post('/api/chat', json={'message': 'Druga'})
        assert one.get('/api/chats').json()[0]['title'] == 'Pierwsza'
    assert all(p.closed == 1 for p in lp + rp)


def test_settings_cannot_store_key_or_change_pinned_endpoint(tmp_path):
    app, _ = setup_app(tmp_path)
    with TestClient(app) as client:
        assert client.put('/api/settings', json={'api_key': 'do-not-save'}).status_code == 400
        assert client.put('/api/settings', json={'ollama_endpoint': 'http://user:secret@host'}).status_code == 400
        assert client.put('/api/settings', json={'theme': 'light'}).status_code == 200
        text = (tmp_path / 'data/settings.json').read_text()
        assert 'do-not-save' not in text
        assert 'private-secret' not in text
        assert 'private-secret' not in client.get('/api/settings').text
        assert client.get('/api/settings').json()['audio_enabled'] is False
        assert client.post('/api/stt/start').status_code == 503


def test_environment_configuration_and_secret_repr(tmp_path):
    config = RuntimeConfig.from_env(environ={
        'NLP_ENDPOINT': 'http://example.test/v1', 'OPENAI_API_KEY': 'secret-key',
        'NLP_REQUEST_TIMEOUT': '7.5', 'NLP_WORKSPACE': str(tmp_path),
        'CHAT_DATA_DIR': str(tmp_path / 'data'), 'NLP_MODEL': 'my-model',
    })
    assert config.request_timeout == 7.5
    assert config.workspace == tmp_path
    assert config.model == 'my-model'
    assert 'secret-key' not in repr(config)


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        RuntimeConfig(request_timeout=timeout)


def test_import_does_not_import_gui_or_audio_dependencies():
    code = '''
import sys
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'webview', 'sounddevice', 'faster_whisper', 'piper', 'numpy', 'onnxruntime'}:
            raise AssertionError('Optional dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from core.api import create_app
app = create_app()
print('headless import OK')
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'headless import OK' in result.stdout


def test_startup_failure_closes_owned_runtime(tmp_path, monkeypatch):
    import core.WebGui as api
    runtime = build_api_runtime(RuntimeConfig(workspace=tmp_path, data_dir=tmp_path / 'data'),
                                provider_factory=FakeProvider, tool_factory=ToolHandle)
    monkeypatch.setattr(api, 'run_cleanup', lambda request: (_ for _ in ()).throw(RuntimeError('cleanup failed')))
    with pytest.raises(RuntimeError, match='cleanup failed'):
        with TestClient(create_app(runtime_factory=lambda: runtime)):
            pass
    assert runtime.provider.closed == 1


def test_workspace_is_bound_without_changing_process_cwd(tmp_path):
    from tools.ToolBuilder import build_tools
    cwd = Path.cwd()
    tools = build_tools(files=True, math=False, web=False, code=False, workspace_dir=tmp_path)
    result = json.loads(tools.execute_tool('create_file', {'path': 'sample.txt', 'content': 'ok'}))
    assert result['success']
    assert (tmp_path / 'sample.txt').read_text() == 'ok'
    outside = json.loads(tools.execute_tool('read_file', {'path': '../outside.txt'}))
    assert outside['success'] is False
    assert Path.cwd() == cwd


def test_ollama_adapter_applies_timeout_and_closes_client(tmp_path):
    from core.ModelProvider import create_provider
    provider = create_provider(RuntimeConfig(workspace=tmp_path, request_timeout=7.5), ToolHandle())
    assert provider.client.timeout == 7.5
    assert provider.client.max_retries == 0
    assert provider.request_timeout == 7.5
    provider.close()
    assert provider.client.is_closed()


def test_injected_runtime_is_not_closed_by_app_owner(tmp_path):
    runtime = build_api_runtime(RuntimeConfig(workspace=tmp_path, data_dir=tmp_path / 'data'),
                                provider_factory=FakeProvider, tool_factory=ToolHandle)
    with TestClient(create_app(runtime=runtime)) as client:
        assert client.get('/health/live').status_code == 200
    assert runtime.provider.closed == 0
    runtime.close()
    runtime.close()
    assert runtime.provider.closed == 1
