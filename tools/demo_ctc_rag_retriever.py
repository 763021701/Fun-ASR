#!/usr/bin/env python3
"""Minimal demo for the standalone CTC-RAG retriever."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hotword import CTCRagRetriever


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone CTC-RAG retriever demo")
    parser.add_argument("audio", help="Path to the input audio")
    parser.add_argument("--hotwords", required=True, help="Hotword file path or inline newline-separated text")
    parser.add_argument("--model_dir", default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--remote_code", default=None)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--ctc_topk", type=int, default=30)
    parser.add_argument("--max_hotwords", type=int, default=32)
    args = parser.parse_args()

    t0 = time.perf_counter()
    retriever = CTCRagRetriever(
        nano_model=args.model_dir,
        nano_remote_code=args.remote_code,
        device=args.device,
    )
    load_model_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    hotword_count = retriever.load_hotwords(args.hotwords)
    load_hotword_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = retriever.retrieve(
        args.audio,
        max_hotwords=args.max_hotwords,
        ctc_topk=args.ctc_topk,
    )
    retrieve_s = time.perf_counter() - t0

    print("=" * 70)
    print("Standalone CTC-RAG Retriever Demo")
    print("=" * 70)
    print(f"Audio:               {args.audio}")
    print(f"Hotwords:            {args.hotwords}")
    print(f"Hotword count:       {hotword_count}")
    print(f"Device:              {args.device}")
    print(f"Model load time:     {load_model_s * 1000.0:.2f} ms")
    print(f"Hotword load time:   {load_hotword_s * 1000.0:.2f} ms")
    print(f"Retrieve time:       {retrieve_s * 1000.0:.2f} ms")
    print(f"Greedy text:         {result.greedy_text}")
    print(f"Integrated text:     {result.integrated_text}")
    print(f"Retrieved hotwords:  {result.retrieved_hotwords}")
    print(f"Radar hotwords:      {result.radar_hotwords}")
    print(f"Extra hotwords:      {result.extra_hotwords}")
    print(f"Hotword scores:      {result.hotword_scores}")

    audio_timings = result.details.get("audio_timings", {})
    rag_timings = result.details.get("timings", {})
    if audio_timings or rag_timings:
        print("\nTiming breakdown:")
        for name, value in audio_timings.items():
            print(f"  audio.{name:<20} {value * 1000.0:9.2f} ms")
        for name, value in rag_timings.items():
            print(f"  rag.{name:<22} {value * 1000.0:9.2f} ms")


if __name__ == "__main__":
    main()
