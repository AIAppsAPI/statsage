"""Universal LLM handler.

Detection order: explicit choice, claude CLI, codex CLI, Anthropic API,
OpenAI compatible API. Every path is optional, callers must handle None,
statsage falls back to template text when no provider is available.

Force a provider with the STATSAGE_LLM environment variable or the
provider argument: claude, codex, anthropic, openai, off.
"""

import json
import os
import shutil
import subprocess
import urllib.request

DEFAULT_TIMEOUT = 120


def _claude_cli_available():
    if shutil.which("claude") is None:
        return False
    creds = os.path.join(os.path.expanduser("~"), ".claude", ".credentials.json")
    if os.path.exists(creds):
        return True
    # API key environments can still run the CLI without a credentials file.
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _codex_cli_available():
    return shutil.which("codex") is not None


def detect_provider(provider=None):
    """Return the provider name that generate() will use, or None."""
    choice = (provider or os.environ.get("STATSAGE_LLM") or "").strip().lower()
    if choice == "off":
        return None
    if choice in ("claude", "codex", "anthropic", "openai"):
        return choice
    if _claude_cli_available():
        return "claude"
    if _codex_cli_available():
        return "codex"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def generate(prompt, system=None, provider=None, timeout=DEFAULT_TIMEOUT):
    """Return generated text, or None if no provider works.

    Never raises on provider failure, statsage must keep working without
    an LLM, so every error path returns None.
    """
    name = detect_provider(provider)
    if name is None:
        return None
    try:
        if name == "claude":
            return _run_cli(["claude", "-p", _merge(system, prompt)], timeout)
        if name == "codex":
            return _run_cli(["codex", "exec", _merge(system, prompt)], timeout)
        if name == "anthropic":
            return _anthropic_api(prompt, system, timeout)
        if name == "openai":
            return _openai_api(prompt, system, timeout)
    except Exception:
        return None
    return None


def generate_json(prompt, system=None, provider=None, timeout=DEFAULT_TIMEOUT):
    """Like generate() but parses the reply as JSON, tolerating code fences."""
    text = generate(prompt, system=system, provider=provider, timeout=timeout)
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except ValueError:
        return None


def _merge(system, prompt):
    if system:
        return system.strip() + "\n\n" + prompt
    return prompt


def _run_cli(args, timeout):
    result = subprocess.run(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
    )
    if result.returncode != 0:
        return None
    text = result.stdout.decode("utf-8", errors="replace").strip()
    return text or None


def _anthropic_api(prompt, system, timeout):
    body = {
        "model": os.environ.get("STATSAGE_ANTHROPIC_MODEL", "claude-sonnet-5"),
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    text = "".join(parts).strip()
    return text or None


def _openai_api(prompt, system, timeout):
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": os.environ.get("STATSAGE_OPENAI_MODEL", "gpt-5.5"),
        "messages": messages,
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", ""),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices", [])
    if not choices:
        return None
    text = (choices[0].get("message", {}).get("content") or "").strip()
    return text or None
