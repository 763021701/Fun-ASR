"""
RAG Hotword Retrieval Demo

Visualize the actual three-stage RAG hotword retrieval pipeline as it runs
inside model.inference(). All displayed results are read directly from the
model's return value — guaranteed to match what was injected into the LLM.

  Stage 1: FastRAG  - inverted index + Numba JIT edit distance (coarse)
  Stage 2: AccuRAG  - fuzzy phoneme weights re-ranking (precise)
  Stage 3: Matching - substring search with word boundary constraints

Usage:
    python raghw_demo.py audio.wav
    python raghw_demo.py audio.wav --hotword_file hot.txt
    python raghw_demo.py audio.wav --hotword_file hot.txt --top_k 30
    python raghw_demo.py audio.wav --vad                          # enable VAD for long audio
    python raghw_demo.py audio.wav --vad --vad_max_segment 30000  # custom VAD segment length (ms)
"""

import sys
import time
import argparse

import torch


def count_hotwords(path):
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for l in f if l.strip() and not l.strip().startswith('#'))


def main():
    parser = argparse.ArgumentParser(description="RAG Hotword Retrieval Demo")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--hotword_file", default="hot.txt",
                        help="Hotword file path (default: hot.txt)")
    parser.add_argument("--language", default="中文", help="Language (default: 中文)")
    parser.add_argument("--model_dir", default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--max_hotwords", type=int, default=10,
                        help="Max hotwords injected into LLM prompt (default: 10)")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Display top-K results per stage (default: 20)")
    parser.add_argument("--vad", action="store_true",
                        help="Enable VAD model (fsmn-vad) for long audio (>30s)")
    parser.add_argument("--vad_max_segment", type=int, default=30000,
                        help="VAD max single segment length in ms (default: 30000)")
    args = parser.parse_args()

    device = (
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    DIV = "=" * 70

    # =================================================================
    # 1. Load ASR model
    # =================================================================
    print(DIV)
    print("RAG Hotword Retrieval Demo")
    print(DIV)

    hw_count = count_hotwords(args.hotword_file)
    print(f"  Audio:        {args.audio}")
    print(f"  Hotword file: {args.hotword_file} ({hw_count} hotwords)")
    print(f"  Language:     {args.language}")
    print(f"  Device:       {device}")
    print(f"  VAD:          {'enabled (fsmn-vad, max=' + str(args.vad_max_segment) + 'ms)' if args.vad else 'disabled'}")

    from funasr import AutoModel

    print(f"\n[Step 1] Loading model: {args.model_dir} ...")
    model_kwargs = dict(
        model=args.model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device=device,
        hub="ms",
    )
    if args.vad:
        model_kwargs["vad_model"] = "fsmn-vad"
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": args.vad_max_segment}
    model = AutoModel(**model_kwargs)

    # =================================================================
    # 2. Run ASR with RAG hotword pipeline
    # =================================================================
    print(f"\n[Step 2] Running ASR with RAG hotword retrieval ...")
    t_start = time.perf_counter()
    res = model.generate(
        input=[args.audio],
        cache={},
        batch_size=1,
        hotword_file=args.hotword_file,
        max_hotwords=args.max_hotwords,
        language=args.language,
        itn=True,
    )
    t_asr = time.perf_counter() - t_start

    n_segments = len(res)
    print(f"  ASR time:   {t_asr:.2f}s  ({n_segments} segment{'s' if n_segments > 1 else ''})")

    # =================================================================
    # 3. Display RAG pipeline results for each segment
    # =================================================================
    # Collect summary stats across all segments
    all_final_texts = []
    total_fast = total_accu = 0

    for seg_idx, result in enumerate(res):
        seg_label = f"Segment {seg_idx + 1}/{n_segments}" if n_segments > 1 else "Result"
        ctc_text = result.get("ctc_text", "")
        final_text = result.get("text", "")
        rag_meta = result.get("rag_meta")
        final_hotwords = result.get("rag_final_hotwords", [])
        all_final_texts.append(final_text)

        print(f"\n{'─' * 70}")
        print(f"[{seg_label}]")
        print(f"{'─' * 70}")
        print(f"  CTC text:   {ctc_text}")

        if not rag_meta:
            print(f"  [!] No RAG metadata — RAG pipeline did not run for this segment.")
            print(f"  Final text: {final_text}")
            continue

        details = rag_meta.get("rag_details") or {}
        correction = rag_meta.get("rag_correction")
        rag_retrieved = rag_meta.get("rag_retrieved_hotwords", [])

        fast_raw = details.get("fast_raw", [])
        accu_raw = details.get("accu_raw", [])
        merged   = details.get("merged", [])
        total_fast += len(fast_raw)
        total_accu += len(accu_raw)

        # Stage 1: FastRAG
        print(f"\n  [Stage 1] FastRAG — {len(fast_raw)} candidates")
        if fast_raw:
            show = min(args.top_k, len(fast_raw))
            for i, (hw, score) in enumerate(fast_raw[:show]):
                print(f"    {i+1:3d}. {hw:<20s}  score={score:.4f}")
            if len(fast_raw) > show:
                print(f"    ... and {len(fast_raw) - show} more")
        else:
            print("    (none)")

        # Stage 2: AccuRAG
        print(f"\n  [Stage 2] AccuRAG — {len(accu_raw)} re-ranked")
        if accu_raw:
            show = min(args.top_k, len(accu_raw))
            for i, (hw, score, start, end) in enumerate(accu_raw[:show]):
                print(f"    {i+1:3d}. {hw:<20s}  score={score:.4f}  pos=[{start}:{end}]")
            if len(accu_raw) > show:
                print(f"    ... and {len(accu_raw) - show} more")
        else:
            print("    (none)")

        # Stage 2+: Merged
        if merged:
            print(f"\n  [Stage 2+] Merged — top {min(args.top_k, len(merged))}")
            show = min(args.top_k, len(merged))
            for i, (hw, score) in enumerate(merged[:show]):
                print(f"    {i+1:3d}. {hw:<20s}  score={score:.4f}")
            if len(merged) > show:
                print(f"    ... and {len(merged) - show} more")

        # Stage 3: Matching
        print(f"\n  [Stage 3] Substring Matching")
        if correction and correction.matchs:
            print(f"    Matched (replaced in text):")
            for orig, hw, score in correction.matchs:
                print(f"      \"{orig}\" -> \"{hw}\"  score={score:.4f}")
        else:
            print(f"    Matched: (none)")
        if correction and correction.similars:
            print(f"    Similar (injected as hotword candidates):")
            for orig, hw, score in correction.similars:
                print(f"      \"{orig}\" ~ \"{hw}\"  score={score:.4f}")
        else:
            print(f"    Similar: (none)")

        print(f"\n  RAG retrieved: {rag_retrieved}")
        print(f"  Final hotwords to LLM: {final_hotwords}")
        print(f"  LLM output: {final_text}")

    # =================================================================
    # 4. Summary
    # =================================================================
    full_text = " ".join(t for t in all_final_texts if t)
    print(f"\n{DIV}")
    print("Summary — What Actually Happened")
    print(DIV)
    print(f"  Hotword pool:    {hw_count} words")
    print(f"  Segments:        {n_segments}")
    print(f"  FastRAG total:   {total_fast} candidates (across all segments)")
    print(f"  AccuRAG total:   {total_accu} re-ranked  (across all segments)")
    print(f"  Total time:      {t_asr:.2f}s")
    print(f"  Full transcript: {full_text}")
    print(DIV)


if __name__ == "__main__":
    main()
