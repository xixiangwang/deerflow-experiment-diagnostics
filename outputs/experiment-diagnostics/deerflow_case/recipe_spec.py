"""Strict data-only contract for small CPU reproductions. No executable fields."""
import copy
import hashlib
import json
import math
from pathlib import Path
import re

MAX_BYTES = 32000


def object_fields(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError("Unknown or invalid fields in " + name)


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(name + " must be an integer in " + str(low) + ".." + str(high))


def normalize_recipe(data):
    object_fields(data, {"schema_version", "name", "profile", "model", "lora", "training", "data"}, "recipe")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if data.get("profile") != "tiny-qwen2-cpu":
        raise ValueError("Only the bounded tiny-qwen2-cpu profile is executable")
    name = data.get("name", "selected-experiment")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("name must use 1..64 ASCII letters, digits, underscores or hyphens")
    model = data.get("model", {})
    object_fields(model, {"vocab_size"}, "model")
    vocab = model.get("vocab_size", 64)
    integer(vocab, 16, 512, "vocab_size")
    lora = data.get("lora", {})
    object_fields(lora, {"r", "alpha", "target_modules"}, "lora")
    rank, alpha = lora.get("r", 2), lora.get("alpha", 4)
    integer(rank, 1, 8, "r")
    integer(alpha, 1, 32, "alpha")
    targets = lora.get("target_modules", ["q_proj", "v_proj"])
    if (not isinstance(targets, list) or not 1 <= len(targets) <= 8 or
        any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", value) for value in targets)
        or len(set(targets)) != len(targets)):
        raise ValueError("target_modules requires 1..8 distinct module identifiers")
    training = data.get("training", {})
    object_fields(training, {"steps", "learning_rate", "gradients_enabled", "collate"}, "training")
    steps, rate = training.get("steps", 2), training.get("learning_rate", .05)
    integer(steps, 2, 3, "steps")
    if type(rate) not in {int, float} or not math.isfinite(rate) or not 0.000001 <= rate <= .1:
        raise ValueError("learning_rate must be finite and within 0.000001..0.1")
    enabled, collate = training.get("gradients_enabled", True), training.get("collate", "pad")
    if type(enabled) is not bool or not isinstance(collate, str) or collate not in {"pad", "stack"}:
        raise ValueError("Invalid gradients_enabled or collate")
    samples = data.get("data", {})
    object_fields(samples, {"input_ids", "labels"}, "data")
    inputs = samples.get("input_ids")
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 4:
        raise ValueError("Provide 1..4 input sequences")
    for sequence in inputs:
        if not isinstance(sequence, list) or not 3 <= len(sequence) <= 64:
            raise ValueError("Input sequence lengths must be 3..64")
        for token in sequence:
            integer(token, 1, vocab - 1, "input token (0 is reserved for padding)")
    labels = samples.get("labels")
    if labels is not None:
        if not isinstance(labels, list) or len(labels) != len(inputs):
            raise ValueError("labels must have one sequence per input")
        for sequence in labels:
            if not isinstance(sequence, list) or not 1 <= len(sequence) <= 64:
                raise ValueError("Label sequence lengths must be 1..64")
            for label in sequence:
                if type(label) is not int or (label != -100 and not 0 <= label <= vocab + 16):
                    raise ValueError("Labels must be -100 or integers in the bounded label range")
    recipe = {"schema_version": 1, "name": name, "profile": "tiny-qwen2-cpu", "model": {"vocab_size": vocab},
        "lora": {"r": rank, "alpha": alpha, "target_modules": list(targets)},
        "training": {"steps": steps, "learning_rate": float(rate), "gradients_enabled": enabled, "collate": collate},
        "data": {"input_ids": copy.deepcopy(inputs), "labels": copy.deepcopy(labels)}}
    if len(canonical_bytes(recipe)) > MAX_BYTES:
        raise ValueError("Recipe exceeds 32000 bytes")
    return recipe


def canonical_bytes(recipe):
    return json.dumps(recipe, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def recipe_hash(recipe):
    return hashlib.sha256(canonical_bytes(recipe)).hexdigest()


def read_recipe(path):
    return load_recipe(path)[0]


def load_recipe(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Select a regular JSON recipe file of at most 32000 bytes")
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("Recipe exceeds 32000 bytes")
    return normalize_recipe(json.loads(raw.decode("utf-8-sig"))), hashlib.sha256(raw).hexdigest()


def apply_change(recipe, change, target_modules=None, observed_modules=()):
    updated = copy.deepcopy(recipe)
    if change == "set_lora_targets":
        if not target_modules or not set(target_modules) <= set(observed_modules) - {"lm_head"}:
            raise ValueError("Select actual observed internal modules")
        updated["lora"]["target_modules"] = target_modules
    elif change == "pad_batch":
        updated["training"]["collate"] = "pad"
    elif change == "enable_gradients":
        updated["training"]["gradients_enabled"] = True
    elif change == "rebuild_causal_labels":
        updated["data"]["labels"] = None
    else:
        raise ValueError("Unsupported recipe change")
    return normalize_recipe(updated)
