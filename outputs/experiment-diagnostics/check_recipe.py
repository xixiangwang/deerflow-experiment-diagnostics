"""Data contract/resource check only; no model API, training, or submitted source execution."""
import argparse
import json
from pathlib import Path
from deerflow_case.recipe_spec import load_recipe, recipe_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    try:
        recipe, source_hash = load_recipe(args.input)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"valid_data_contract": True, "profile": recipe["profile"],
        "input_transport_sha256": source_hash, "recipe_sha256": recipe_hash(recipe),
        "sample_lengths": list(map(len, recipe["data"]["input_ids"])), "steps": recipe["training"]["steps"],
        "device": "cpu", "pretrained_weights_used": False, "validation_executed": False,
        "notice": "数据格式和资源范围检查通过，不证明模型配置正确或实验可运行。"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
