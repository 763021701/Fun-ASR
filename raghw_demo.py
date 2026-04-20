"""
CTC-RAG Hotword Retrieval Demo

Visualize the actual CTC-RAG hotword retrieval pipeline as it runs inside
model.inference(). All displayed results are read directly from the model's
return value and reflect what was injected into the LLM.

  Stage 1: Radar Scan         - search hotwords in the Top-K CTC lattice
  Stage 2: Integrated CTC     - merge radar hits into the greedy CTC stream
  Stage 3: Phoneme Correction - fuzzy correction and extra hotword expansion

Usage:
    python raghw_demo.py audio.wav
    python raghw_demo.py audio.wav --vad  # enable VAD
    python raghw_demo.py audio.wav --vad --hotwords hot.txt --mode prompt  # prompt mode
    python raghw_demo.py audio.wav --vad --hotwords hot.txt --mode ctc_rag --max_hotwords 30 --top_k 30
"""

import sys
import time
import argparse

import torch


def count_hotwords(path):
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for l in f if l.strip() and not l.strip().startswith('#'))


def main():
    parser = argparse.ArgumentParser(description="CTC-RAG Hotword Retrieval Demo")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--hotwords", default=None,
                        help="Hotword file path or comma-separated list (default: disabled)")
    parser.add_argument("--mode", default="ctc_rag", choices=["ctc_rag", "prompt"],
                        help="Hotword mode: 'ctc_rag' (default) or 'prompt'")
    parser.add_argument("--language", default=None,
                        help="Language for ASR prompt (default: None)")
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
    print("CTC-RAG Hotword Retrieval Demo")
    print(DIV)

    import os
    use_hotwords = bool(args.hotwords)
    is_file = bool(args.hotwords) and os.path.isfile(args.hotwords)
    hw_count = 0
    if use_hotwords:
        hw_count = count_hotwords(args.hotwords) if is_file else len(args.hotwords.split(","))
    print(f"  Audio:        {args.audio}")
    if use_hotwords:
        print(f"  Hotwords:     {args.hotwords} ({hw_count} hotwords)")
        print(f"  Mode:         {args.mode}")
    else:
        print("  Hotwords:     disabled")
        print("  Mode:         normal")
    lang_display = args.language if args.language is not None else "None (model default)"
    print(f"  Language:     {lang_display}")
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
    if use_hotwords:
        print(f"\n[Step 2] Running ASR with CTC-RAG hotword retrieval ...")
    else:
        print(f"\n[Step 2] Running ASR without hotwords ...")
    t_start = time.perf_counter()
    generate_kwargs = dict(
        input=[args.audio],
        cache={},
        batch_size=1,
        language=args.language,
        itn=True,
    )
    if use_hotwords:
        hotwords_val = args.hotwords if is_file else args.hotwords.split(",")
        generate_kwargs["hotwords"] = hotwords_val
        generate_kwargs["hotword_mode"] = args.mode
        generate_kwargs["max_hotwords"] = args.max_hotwords
    res = model.generate(**generate_kwargs)
    t_asr = time.perf_counter() - t_start

    print(f"  ASR time:   {t_asr:.2f}s")

    # =================================================================
    # 3. Display CTC-RAG pipeline results
    # =================================================================
    # With VAD, funasr merges all segment results into one result dict.
    # rag_meta is a list of per-segment dicts, rag_final_hotwords is a list of per-segment lists.
    all_final_texts = []
    total_radar = total_extra = 0
    total_vad_segments = 0
    global_hw_scores = {}  # {hw: best_score} across all segments

    for result in res:
        final_text = result.get("text", "")
        all_final_texts.append(final_text)

        rag_meta_list = result.get("rag_meta") or []
        final_hws_list = result.get("rag_final_hotwords") or []
        n_vad_segments = len(rag_meta_list)
        total_vad_segments += n_vad_segments

        if n_vad_segments == 0:
            if use_hotwords:
                print(f"\n  [!] No CTC-RAG metadata — retrieval pipeline did not run.")
            else:
                print(f"\n  [!] Normal inference mode — no retrieval metadata.")
            print(f"  Final text: {final_text}")
            continue

        print(f"\n  VAD segments with retrieval metadata: {n_vad_segments}")

        for seg_i, rag_meta in enumerate(rag_meta_list):
            seg_hws = final_hws_list[seg_i] if seg_i < len(final_hws_list) else []
            ctc_text = rag_meta.get("rag_ctc_text", "")
            integrated_text = rag_meta.get("rag_integrated_text", ctc_text)
            details = rag_meta.get("rag_details") or {}
            correction = rag_meta.get("rag_correction")
            rag_retrieved = rag_meta.get("rag_retrieved_hotwords", [])
            radar_texts = rag_meta.get("rag_radar_texts", [])
            extra_hotwords = rag_meta.get("rag_extra_hotwords", [])
            ctc_topk = rag_meta.get("rag_ctc_topk")
            ctc_rag_timings = rag_meta.get("rag_ctc_rag_timings") or {}
            total_radar += len(radar_texts)
            total_extra += len(extra_hotwords)

            # Track global hotword scores
            for hw in rag_retrieved:
                if hw not in global_hw_scores:
                    global_hw_scores[hw] = 0.0
            if correction:
                for _, hw, score in correction.matchs:
                    global_hw_scores[hw] = max(global_hw_scores.get(hw, 0), score)
                for _, hw, score in correction.similars:
                    global_hw_scores[hw] = max(global_hw_scores.get(hw, 0), score)

            # Per-segment header
            print(f"\n{'─' * 70}")
            print(f"  [Segment {seg_i + 1}/{n_vad_segments}]")
            print(f"{'─' * 70}")
            print(f"  Greedy CTC text: {ctc_text}")
            if integrated_text != ctc_text:
                print(f"  Integrated CTC text: {integrated_text}")

            print(f"\n  [Stage 1] Radar Scan — {len(radar_texts)} detected hotwords (top_k={ctc_topk})")
            if radar_texts:
                show = min(args.top_k, len(radar_texts))
                for i, hw in enumerate(radar_texts[:show]):
                    print(f"    {i+1:3d}. {hw}")
                if len(radar_texts) > show:
                    print(f"    ... and {len(radar_texts) - show} more")
            else:
                print("    (none)")

            print(f"\n  [Stage 2] Integrated CTC")
            if integrated_text != ctc_text:
                print("    Radar hits were merged into the greedy token stream.")
            else:
                print("    No radar merge was applied.")

            print(f"\n  [Stage 3] Phoneme Correction")
            if correction and correction.matchs:
                print("    Matched (replaced in text):")
                for orig, hw, score in correction.matchs:
                    print(f"      \"{orig}\" -> \"{hw}\"  score={score:.4f}")
            else:
                print("    Matched: (none)")
            if correction and correction.similars:
                print("    Similar (injected as hotword candidates):")
                for orig, hw, score in correction.similars:
                    print(f"      \"{orig}\" ~ \"{hw}\"  score={score:.4f}")
            else:
                print("    Similar: (none)")
            if extra_hotwords:
                print(f"    Extra hotwords from correction: {extra_hotwords[:args.top_k]}")
                if len(extra_hotwords) > args.top_k:
                    print(f"    ... and {len(extra_hotwords) - args.top_k} more")

            if ctc_rag_timings:
                print("\n  Timings:")
                for name, value in ctc_rag_timings.items():
                    print(f"    {name:<18s} {value * 1000.0:8.2f} ms")

            print(f"\n  Hotwords to LLM: {seg_hws}")

    # =================================================================
    # 4. Summary
    # =================================================================
    full_text = " ".join(t for t in all_final_texts if t)
    ranked_global = sorted(global_hw_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{DIV}")
    print("Summary — What Actually Happened")
    print(DIV)
    print(f"  Hotword pool:       {hw_count} words")
    print(f"  Hotword mode:       {args.mode if use_hotwords else 'normal'}")
    print(f"  VAD segments:       {total_vad_segments}")
    print(f"  Radar hits total:   {total_radar} (across all segments)")
    print(f"  Correction extras:  {total_extra} (across all segments)")
    print(f"  Unique hotwords:    {len(global_hw_scores)} (across all segments)")
    if ranked_global:
        print(f"  All retrieved hotwords (by best score):")
        for i, (hw, score) in enumerate(ranked_global):
            print(f"    {i+1:3d}. {hw:<20s}  best_score={score:.4f}")
    print(f"  Total time:         {t_asr:.2f}s")
    print(f"  Full transcript:    {full_text}")
    print(DIV)


if __name__ == "__main__":
    main()
