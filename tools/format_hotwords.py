"""
Format hotword file into one hotword per line with deduplication.

Usage:
    python tools/format_hotwords.py input.txt
    python tools/format_hotwords.py input.txt -o output.txt
    python tools/format_hotwords.py input.txt --inplace
    python tools/format_hotwords.py input.txt --lang en
"""

import argparse
import sys
from pathlib import Path


def normalize_hotword(token: str, lang: str) -> str:
    token = token.strip()
    if not token:
        return ""
    if lang == "en":
        token = token.replace("_", " ")
        token = " ".join(token.split())
    return token


def collect_hotwords(input_path: Path, lang: str) -> list[str]:
    seen = set()
    result = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            for token in stripped.split():
                word = normalize_hotword(token, lang)
                if not word or word in seen:
                    continue
                seen.add(word)
                result.append(word)

    return result


def main():
    parser = argparse.ArgumentParser(description="Format hotword file to one-per-line")
    parser.add_argument("input", help="Input hotword file")
    parser.add_argument("-o", "--output", help="Output file (default: print to stdout)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite input file in-place")
    parser.add_argument(
        "--lang",
        choices=["zh", "en"],
        default="zh",
        help="Input keyword style: zh keeps tokens as-is; en also converts '_' to spaces",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    result = collect_hotwords(input_path, args.lang)

    output_text = "\n".join(result) + "\n"

    if args.inplace:
        with input_path.open('w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Formatted {len(result)} hotwords -> {args.input}")
    elif args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Formatted {len(result)} hotwords -> {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output_text)
        print(f"Total: {len(result)} hotwords", file=sys.stderr)


if __name__ == "__main__":
    main()
