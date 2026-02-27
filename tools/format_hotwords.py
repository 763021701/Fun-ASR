"""
Format hotword file: split space-separated hotwords into one per line, deduplicate.

Usage:
    python tools/format_hotwords.py input.txt
    python tools/format_hotwords.py input.txt -o output.txt
    python tools/format_hotwords.py input.txt --inplace
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Format hotword file to one-per-line")
    parser.add_argument("input", help="Input hotword file")
    parser.add_argument("-o", "--output", help="Output file (default: print to stdout)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite input file in-place")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    seen = set()
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        for word in stripped.split():
            if word not in seen:
                seen.add(word)
                result.append(word)

    output_text = "\n".join(result) + "\n"

    if args.inplace:
        with open(args.input, 'w', encoding='utf-8') as f:
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
