import os

from statsage import providers


def test_off_disables_everything(monkeypatch):
    monkeypatch.setenv("STATSAGE_LLM", "off")
    assert providers.detect_provider() is None
    assert providers.generate("hello") is None


def test_explicit_argument_wins(monkeypatch):
    monkeypatch.delenv("STATSAGE_LLM", raising=False)
    assert providers.detect_provider("off") is None
    assert providers.detect_provider("anthropic") == "anthropic"


def test_env_key_detection(monkeypatch):
    monkeypatch.setenv("STATSAGE_LLM", "")
    monkeypatch.setattr(providers, "_claude_cli_available", lambda: False)
    monkeypatch.setattr(providers, "_codex_cli_available", lambda: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert providers.detect_provider() is None
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert providers.detect_provider() == "openai"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert providers.detect_provider() == "anthropic"


def test_generate_json_tolerates_fences():
    assert providers.generate_json is not None
    # direct parse path
    import statsage.providers as p
    original = p.generate
    try:
        p.generate = lambda *a, **k: "```json\n{\"a\": 1}\n```"
        assert p.generate_json("x") == {"a": 1}
        p.generate = lambda *a, **k: "no json here"
        assert p.generate_json("x") is None
    finally:
        p.generate = original
