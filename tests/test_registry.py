from pathlib import Path

from wildlife_csi.registry import describe_registry, load_registry


def test_registry_has_gpt6_sol():
    reg = load_registry(Path(__file__).parents[1] / "configs/models")
    assert "gpt-6-sol" in reg
    assert {
        "astra",
        "deepseek",
        "fable",
        "gemini",
        "glm",
        "grok",
        "luna",
        "muse",
        "opus",
    } <= set(reg)
    assert "sol" not in reg and "terra" not in reg
    cfg = reg["gpt-6-sol"]
    assert cfg["model"] == "gpt-6-sol"
    assert cfg["base_url_env"] == "GPT6_SOL_BASE_URL"
    assert reg["opus"]["adapter"] == "bedrock-converse"
    assert reg["opus"]["model"] == "global.anthropic.claude-opus-5-5"
    assert reg["opus"]["api_key_env"] == "OPUS55_API_KEY"


def test_describe_safe_no_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("GPT6_SOL_API_KEY", raising=False)
    rows = describe_registry(Path(__file__).parents[1] / "configs/models")
    row = next(r for r in rows if r["id"] == "gpt-6-sol")
    assert "API_KEY" not in str(row.get("base_url", ""))
    assert row["has_key"] is False
