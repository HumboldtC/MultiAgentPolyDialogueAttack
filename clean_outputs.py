import argparse
import json
from pathlib import Path


def is_bad_record(line: str, keyword: str) -> bool:
    if keyword in line:
        return True
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return True
    rounds = obj.get("rounds", [])
    for rnd in rounds:
        for key, value in rnd.items():
            if key.startswith("answer") and isinstance(value, str) and keyword in value:
                return True
    return False


def clean_file(path: Path, keyword: str) -> int:
    good_lines = []
    bad_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if is_bad_record(line, keyword):
                bad_count += 1
                continue
            good_lines.append(line)
    if bad_count > 0:
        path.write_text("\n".join(good_lines) + ("\n" if good_lines else ""), encoding="utf-8")
    return bad_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="jsonl文件或目录")
    parser.add_argument("--keyword", default="API调用异常", help="判定为错误的关键词")
    args = parser.parse_args()

    root = Path(args.path)
    if root.is_file():
        bad = clean_file(root, args.keyword)
        print(f"{root}: removed {bad}")
        return

    if root.is_dir():
        total_bad = 0
        for file in root.rglob("*.jsonl"):
            total_bad += clean_file(file, args.keyword)
        print(f"{root}: removed {total_bad}")
        return

    raise SystemExit(f"path not found: {root}")


if __name__ == "__main__":
    main()
