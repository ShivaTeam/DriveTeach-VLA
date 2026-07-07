"""Write QA pairs to JSON or JSONL format."""

import json
import os


def write_output(qa_pairs: list[dict], output_cfg: dict):
    """Write QA pairs to file.

    Args:
        qa_pairs: list of {id, image, system, conversations} dicts
        output_cfg: {format: "json"|"jsonl", path: str, chunk_size: int|null}
    """
    fmt = output_cfg.get("format", "json")
    path = output_cfg["path"]
    chunk_size = output_cfg.get("chunk_size")

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    if fmt == "jsonl":
        _write_jsonl(qa_pairs, path)
    else:
        if chunk_size:
            _write_json_chunked(qa_pairs, path, chunk_size)
        else:
            _write_json(qa_pairs, path)


def _write_json(data: list[dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_json_chunked(data: list[dict], base_path: str, chunk_size: int):
    base, ext = os.path.splitext(base_path)
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_path = f"{base}_{i // chunk_size + 1}{ext}"
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        print(f"  Wrote chunk {i // chunk_size + 1}: {len(chunk)} samples → {chunk_path}")


def _write_jsonl(data: list[dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
