"""Read-only shape checks for the one supported local pretrained experiment profile."""
import hashlib
import json
from pathlib import Path

EXPECTED = {"model_type": "qwen2", "hidden_size": 896, "intermediate_size": 4864,
    "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2,
    "vocab_size": 151936, "tie_word_embeddings": True}


def inspect_local_profile(directory):
    directory = Path(directory)
    path = directory / "config.json"
    if not path.is_file() or path.stat().st_size > 32768:
        raise ValueError("Expected a local config.json of at most 32768 bytes")
    raw = path.read_bytes()
    if len(raw) > 32768:
        raise ValueError("Oversized model config")
    config = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(config, dict) or "auto_map" in config:
        raise ValueError("Custom model code mappings are unsupported")
    mismatched = [key for key, value in EXPECTED.items() if type(config.get(key)) is not type(value) or config.get(key) != value]
    if mismatched or config.get("architectures") != ["Qwen2ForCausalLM"]:
        raise ValueError("Only the Qwen2.5-0.5B configuration is allowed; unsupported fields: " + ", ".join(mismatched))
    weight = directory / "model.safetensors"
    if not weight.is_file() or not 1000 < weight.stat().st_size <= 2500000000:
        raise ValueError("Expected one existing 0.5B model.safetensors file; other weight formats/shards are unsupported")
    return {"profile": "local-qwen2.5-0.5b", "config_sha256": hashlib.sha256(raw).hexdigest(),
        "checked_dimensions": EXPECTED.copy(), "weight_file_bytes": weight.stat().st_size,
        "weight_contents_verified": False, "pretrained_weights_loaded_by_preflight": False}
