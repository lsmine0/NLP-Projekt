from core import Logger


def test_file_logging_records_levels_without_thinking(tmp_path, monkeypatch):
    monkeypatch.delenv("NLP_LOG_FILE", raising=False)
    path = Logger.configure_file_logging(tmp_path / "logs" / "app.log")
    assert path is not None
    old_verbose = Logger.VERBOSE
    Logger.VERBOSE = 0
    try:
        Logger.printInfo("info message")
        Logger.printWarning("warning message")
        Logger.printError("error message")
        Logger.printThinking("reasoning should stay in memory")
    finally:
        Logger.VERBOSE = old_verbose
        Logger.configure_file_logging(None)

    text = path.read_text(encoding="utf-8")
    assert "INFO info message" in text
    assert "WARNING warning message" in text
    assert "ERROR error message" in text
    assert "reasoning should stay in memory" not in text
    assert (path.stat().st_mode & 0o777) == 0o600


def test_env_log_file_override(tmp_path, monkeypatch):
    target = tmp_path / "custom.log"
    monkeypatch.setenv("NLP_LOG_FILE", str(target))
    try:
        assert Logger.configure_file_logging(tmp_path / "ignored.log") == target
        Logger.printInfo("env override")
    finally:
        monkeypatch.delenv("NLP_LOG_FILE", raising=False)
        Logger.configure_file_logging(None)
    assert "env override" in target.read_text(encoding="utf-8")
