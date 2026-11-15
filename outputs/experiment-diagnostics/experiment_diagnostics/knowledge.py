"""Version-filtered lexical retrieval. BM25 scores are not probabilities."""
import hashlib
import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from .legacy_rag import chunk_semantic, route_by_rule

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = Path(os.environ.get("DIAG_KNOWLEDGE_INDEX", ROOT / "knowledge" / "index.sqlite3"))
ALIASES = {
    "显存": "CUDA memory out of memory allocation batch size",
    "量化": "quantization bitsandbytes nf4 load_in_4bit",
    "微调": "finetuning training LoRA adapter",
    "梯度": "gradient backward requires_grad",
    "依赖": "installation package version import",
    "填充": "padding pad_token tokenizer",
    "模板": "chat template apply_chat_template",
    "随机": "randomness seed reproducibility",
    "混合精度": "automatic mixed precision autocast GradScaler",
    "保存": "checkpoint save_pretrained serialization",
    "非有限": "nan inf numerical accuracy overflow",
}


def tokens(text):
    values = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower())
    out = []
    for word in values:
        out.append(word)
        if "_" in word:
            out.extend(word.split("_"))
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            out.extend(word[i:i + 2] for i in range(len(word) - 1))
    return " ".join(out)


def build_index(root=ROOT, destination=DEFAULT_INDEX):
    root, destination = Path(root), Path(destination)
    manifest = json.loads((root / "knowledge/manifest.json").read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".building.sqlite3")
    if temporary.exists():
        temporary.unlink()
    count = 0
    with closing(sqlite3.connect(temporary)) as db:
        db.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, package TEXT, version TEXT, url TEXT, source_sha256 TEXT, source_line_start INTEGER, source_line_end INTEGER, text TEXT)")
        db.execute("CREATE VIRTUAL TABLE search USING fts5(id UNINDEXED, terms)")
        for doc in manifest:
            if doc["status"] != "ok":
                continue
            source = (root / doc["file"]).resolve()
            if not source.is_relative_to(root.resolve()):
                raise ValueError("Document path escapes knowledge root")
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != doc["sha256"]:
                raise ValueError("Source digest mismatch: " + doc["id"])
            text = raw.decode("utf-8")
            cursor = 0
            for chunk in chunk_semantic(text, doc["id"]):
                # Find exact text. Skip transformed chunks rather than invent line citations.
                position = text.find(chunk["text"], cursor)
                if position < 0:
                    position = text.find(chunk["text"])
                if position < 0:
                    continue
                cursor = position + len(chunk["text"])
                first = text.count("\n", 0, position) + 1
                last = text.count("\n", 0, cursor) + 1
                identifier = doc["id"] + ":" + str(chunk["chunk_id"])
                db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)", (
                    identifier, doc["package"], doc["version"], doc["url"], doc["sha256"],
                    first, last, chunk["text"]))
                db.execute("INSERT INTO search VALUES(?,?)", (identifier, tokens(chunk["text"])))
                count += 1
        db.commit()
    temporary.replace(destination)
    return {"documents": sum(d["status"] == "ok" for d in manifest),
            "chunks": count, "index": str(destination), "method": "SQLite FTS5 BM25"}


def retrieve(query, top_k=3, versions=None, package=None, index=DEFAULT_INDEX, diversify_sources=False):
    if not isinstance(query, str) or not query.strip() or len(query) > 10000:
        raise ValueError("query must have 1..10000 characters")
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer in 1..10")
    if type(diversify_sources) is not bool:
        raise ValueError("diversify_sources must be boolean")
    if versions is not None and (not isinstance(versions, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in versions.items())):
        raise ValueError("versions must map package names to version strings")
    if not route_by_rule(query) and re.fullmatch(r"(?:你好|您好|谢谢|感谢|再见|拜拜|在吗|辛苦了)[！!。\s]*", query.strip()):
        return {"status": "skipped_chitchat", "hits": []}
    if not Path(index).exists():
        return {"status": "index_unavailable", "hits": []}
    expanded = query + " " + " ".join(v for k, v in ALIASES.items() if k in query)
    terms = list(dict.fromkeys(tokens(expanded).split()))[:80]
    if not terms:
        return {"status": "insufficient_evidence", "hits": []}
    match = " OR ".join('"' + term + '"' for term in terms)
    filters, args = [], [match]
    if package:
        filters.append("c.package=?")
        args.append(package)
    # Compare exact package version (ignoring local build suffix such as +cu121).
    versions = versions or {}
    normalized = {("pytorch" if k == "torch" else k): v.split("+")[0].removeprefix("v")
                  for k, v in versions.items()}
    for name, version in normalized.items():
        filters.append("(c.package<>? OR c.version=?)")
        args.extend([name, version])
    where = " AND " + " AND ".join(filters) if filters else ""
    sql = ("SELECT c.*, bm25(search) AS rank FROM search JOIN chunks c ON search.id=c.id "
           "WHERE search MATCH ?" + where + " ORDER BY rank, c.id LIMIT ?")
    args.append(min(200, top_k * 20) if diversify_sources else top_k)
    with closing(sqlite3.connect(Path(index).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        hits = [dict(row) for row in db.execute(sql, args)]
    if diversify_sources:
        unique, remaining, sources = [], [], set()
        for hit in hits:
            source = hit["source_sha256"]
            if source in sources:
                remaining.append(hit)
            else:
                unique.append(hit)
                sources.add(source)
        hits = (unique + remaining)[:top_k]
    for hit in hits:
        hit["citation"] = hit["url"] + f"#L{hit['source_line_start']}-L{hit['source_line_end']}"
        hit["applicability"] = "exact_version" if hit["package"] in normalized else "version_not_verified"
        hit["score"] = -hit.pop("rank")
    return {"status": "retrieved" if hits else "insufficient_evidence", "hits": hits,
            "method": "lexical_bm25_source_diversity" if diversify_sources else "lexical_bm25",
            "notice": "检索相关不等于能够回答；分数不是置信概率。版本未提供的证据需进一步核对。"}


def augment_diagnosis(payload, index=DEFAULT_INDEX):
    from .diagnose import diagnose
    data = dict(payload)
    data["config"] = data.pop("experiment_config", data.get("config", {}))
    report = diagnose(data)
    versions = data.get("environment", {}).get("versions", {})
    queries = {
        "cuda_oom": "CUDA memory management out of memory allocation batch size",
        "missing_module": "installation dependencies package import version",
        "nonfinite_loss": "numerical accuracy inf nan mixed precision overflow GradScaler",
        "invalid_batch_size": "per_device_train_batch_size gradient_accumulation_steps training",
        "missing_file": "checkpoint data file path save load serialization",
        "target_out_of_bounds": "CrossEntropyLoss target labels ignore_index",
        "no_grad": "autograd requires_grad backward no_grad",
        "shape_mismatch": "padding truncation data collator tensors batch",
    }
    report["knowledge"] = []
    for code in dict.fromkeys(item["code"] for item in report["findings"]):
        report["knowledge"].append({"finding": code, **retrieve(queries[code], versions=versions, index=index)})
    report["mode"] = "offline_rules_with_retrieval"
    return report


if __name__ == "__main__":
    print(json.dumps(build_index(), ensure_ascii=False, indent=2))
