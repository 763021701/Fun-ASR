#!/usr/bin/env python3
"""
Compare hotword *retrieval* (candidate list) between:
  (A) Current Fun-ASR repo: CTC-RAG path in model.py
      (Top-K lattice -> Radar -> Integrator -> PhonemeCorrector)
  (B) Reference Fun-ASR-GGUF: CTCDecoder
      (Top-K lattice -> Radar -> Integrator -> PhonemeCorrector)

Inputs:
  --audio       WAV (or format supported by loaders)
  --hots        Hotword library file (one phrase per line, # comments ok)
  --hot_ref     Ground-truth file: hotwords that actually appear in the audio (one per line)

Metrics (against hot_ref):
  - recall  = |retrieved ∩ hot_ref| / |hot_ref|
  - precision_on_ref = |retrieved ∩ hot_ref| / |retrieved|  (undefined if retrieved empty -> 0.0)

Also prints wall time: model load (from_pretrained); reference ORT session build vs inference-only;
per-side retrieval path; slow/fast ratio (reference inference excludes Encoder/CTC ctor+warmup).

Example:
  python tools/compare_hotword_retrieval.py \
    --audio test_files/audio.wav --hots hots.txt --hot_ref hot_ref.txt

Reference side needs ONNX under refs/Fun-ASR-GGUF/model/ or explicit paths.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import time
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_text_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines


def _metrics(hot_ref: Set[str], retrieved: List[str]) -> Dict[str, object]:
    ret_set = set(retrieved)
    hits = hot_ref & ret_set
    missed = hot_ref - ret_set
    extra = ret_set - hot_ref
    n_ref = len(hot_ref)
    n_ret = len(ret_set)
    recall = (len(hits) / n_ref) if n_ref else 0.0
    precision = (len(hits) / n_ret) if n_ret else 0.0
    return {
        "hits": sorted(hits),
        "missed": sorted(missed),
        "extra": sorted(extra),
        "recall": recall,
        "precision_on_ref": precision,
        "retrieved_ordered": list(retrieved),
        "retrieved_unique_count": n_ret,
    }


def _print_side(
    title: str,
    m: Dict[str, object],
    ctc_text: str,
    elapsed_s: float | None = None,
    *,
    elapsed_label: str = "Wall time (retrieval path only):",
    stage_timings: Dict[str, Any] | None = None,
) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    if elapsed_s is not None:
        print(f"{elapsed_label} {elapsed_s * 1000.0:.2f} ms ({elapsed_s:.4f} s)")
    print(f"CTC text (for retrieval): {ctc_text[:200]}{'...' if len(ctc_text) > 200 else ''}")
    print(f"Retrieved ({len(m['retrieved_ordered'])} ordered, {m['retrieved_unique_count']} unique):")
    for i, w in enumerate(m["retrieved_ordered"], 1):
        print(f"  {i:3d}  {w!r}")
    print(f"Recall@hot_ref:    {m['recall']:.4f}")
    print(f"Precision (hits/retrieved): {m['precision_on_ref']:.4f}")
    print(f"Hits ({len(m['hits'])}):   {m['hits']}")
    print(f"Missed ref: {m['missed']}")
    print(f"Extra (not in hot_ref): {m['extra']}")
    if stage_timings:
        _print_stage_timings(stage_timings)


def _print_stage_timings(stage_timings: Dict[str, Any], indent: str = "  ") -> None:
    print(f"{indent}Stage timing breakdown:")
    for name, value in stage_timings.items():
        if isinstance(value, dict):
            print(f"{indent}  {name}:")
            for sub_name, sub_value in value.items():
                print(f"{indent}    {sub_name:<24} {sub_value * 1000.0:9.2f} ms ({sub_value:.4f} s)")
        else:
            print(f"{indent}  {name:<26} {value * 1000.0:9.2f} ms ({value:.4f} s)")


def _run_current_ctc_rag(
    model,
    audio_path: Path,
    hots_path: Path,
    max_hotwords: int,
    ctc_topk: int,
    prep_kwargs: dict,
) -> Tuple[List[str], str, Dict[str, Any]]:
    import os as _os

    from hotword.ctc_rag_hotword import run_ctc_rag_hotword_pipeline

    t0 = time.perf_counter()
    hotword_list = model._load_hotword_lines_from_file(str(hots_path))
    corrector = model._get_hotword_corrector(hotword_list, source_path=_os.path.abspath(str(hots_path)))
    t_hotword_setup = time.perf_counter() - t0
    if corrector is None or not corrector.hotwords:
        return [], "", {"hotword_setup": t_hotword_setup}

    t0 = time.perf_counter()
    prompt_p1 = model.get_prompt([], prep_kwargs.get("language", None), prep_kwargs.get("itn", True))
    data_in_p1 = [model.generate_chatml(prompt_p1, str(audio_path))]
    t_prompt = time.perf_counter() - t0
    key = ["compare_hotword_retrieval_ctc_rag"]
    kw = dict(prep_kwargs)
    tokenizer = kw.pop("tokenizer", None)
    frontend = kw.pop("frontend", None)
    t0 = time.perf_counter()
    _, _, _, _, meta_data = model.inference_prepare(
        data_in_p1,
        None,
        key,
        tokenizer,
        frontend,
        **kw,
    )
    t_prepare = time.perf_counter() - t0
    encoder_out = meta_data["encoder_out"]
    encoder_out_lens = meta_data["encoder_out_lens"]

    t0 = time.perf_counter()
    decoder_out, _ = model.ctc_decoder(encoder_out, encoder_out_lens)
    ctc_logits = model.ctc.log_softmax(decoder_out)
    t_ctc_forward = time.perf_counter() - t0

    t0 = time.perf_counter()
    integrated_text, rag_retrieved, ctc_rag_meta = run_ctc_rag_hotword_pipeline(
        log_probs=ctc_logits[0, : int(encoder_out_lens[0].item()), :],
        blank_id=model.blank_id,
        hotword_lines=hotword_list,
        ctc_tokenizer=model.ctc_tokenizer,
        corrector=corrector,
        max_hotwords=max_hotwords,
        ctc_topk=ctc_topk,
    )
    t_ctc_rag = time.perf_counter() - t0

    load_audio_s = float(meta_data.get("load_data", 0.0) or 0.0)
    extract_feat_s = float(meta_data.get("extract_feat", 0.0) or 0.0)
    prepare_other_est = max(t_prepare - load_audio_s - extract_feat_s, 0.0)
    timings = {
        "hotword_setup": t_hotword_setup,
        "prompt_build": t_prompt,
        "prepare_total": t_prepare,
        "ctc_forward": t_ctc_forward,
        "ctc_rag_total": t_ctc_rag,
        "prepare_detail_est": {
            "load_audio": load_audio_s,
            "extract_feat": extract_feat_s,
            "encode+adaptor+embed_prep_est": prepare_other_est,
        },
        "ctc_rag_detail": dict(ctc_rag_meta.get("timings", {})),
    }
    return rag_retrieved, integrated_text, timings


def _bootstrap_ref_inference(ref_root: Path) -> None:
    if "fun_asr_gguf.inference.ctc_decoder" in sys.modules:
        return
    gguf_pkg = ref_root / "fun_asr_gguf"
    inf_dir = gguf_pkg / "inference"
    hw_dir = inf_dir / "hotword"
    log = logging.getLogger("compare_hotword_ref")
    log.setLevel(logging.WARNING)
    if not log.handlers:
        log.addHandler(logging.StreamHandler())

    if "fun_asr_gguf" not in sys.modules:
        m_top = types.ModuleType("fun_asr_gguf")
        m_top.__path__ = [str(gguf_pkg)]
        sys.modules["fun_asr_gguf"] = m_top
    m_inf = types.ModuleType("fun_asr_gguf.inference")
    m_inf.__path__ = [str(inf_dir)]
    m_inf.logger = log
    sys.modules["fun_asr_gguf.inference"] = m_inf
    m_hw = types.ModuleType("fun_asr_gguf.inference.hotword")
    m_hw.__path__ = [str(hw_dir)]
    m_hw.logger = log
    sys.modules["fun_asr_gguf.inference.hotword"] = m_hw

    def _load(full_name: str, file_path: Path) -> None:
        spec = importlib.util.spec_from_file_location(full_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {full_name} from {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)

    _load("fun_asr_gguf.inference.radar", inf_dir / "radar.py")
    _load("fun_asr_gguf.inference.integrator", inf_dir / "integrator.py")
    _load("fun_asr_gguf.inference.hotword.algo_phoneme", hw_dir / "algo_phoneme.py")
    _load("fun_asr_gguf.inference.hotword.algo_calc", hw_dir / "algo_calc.py")
    _load("fun_asr_gguf.inference.hotword.rag_fast", hw_dir / "rag_fast.py")
    _load("fun_asr_gguf.inference.hotword.hot_phoneme", hw_dir / "hot_phoneme.py")
    _load("fun_asr_gguf.inference.encoder", inf_dir / "encoder.py")
    _load("fun_asr_gguf.inference.ctc_decoder", inf_dir / "ctc_decoder.py")


def _pick_onnx(model_dir: Path, stems: List[str]) -> Path:
    for stem in stems:
        for ext in (".int4.onnx", ".fp16.onnx", ".fp32.onnx"):
            p = model_dir / f"{stem}{ext}"
            if p.exists():
                return p
    tried = [str(model_dir / f"{s}{e}") for s in stems for e in (".int4.onnx", ".fp16.onnx", ".fp32.onnx")]
    raise FileNotFoundError("No ONNX found. Tried:\n  " + "\n  ".join(tried))


def _resolve_onnx_provider(cli: str | None, pytorch_device: str) -> str:
    """
    Match ORT device class to PyTorch when possible.
    AUTO -> CUDA if current runs on CUDA and torch sees a GPU, else CPU.
    """
    if cli:
        u = cli.strip().upper()
        if u != "AUTO":
            return u
    if str(pytorch_device).lower().startswith("cuda") and torch.cuda.is_available():
        return "CUDA"
    return "CPU"


def _cap_reference_hotword_list(words: List[str], max_n: int) -> List[str]:
    """
    Reference CTCDecoder returns radar hits ∪ phoneme-corrector candidates without a hard
    cap like current RAG. Optional cap: dedupe (stable), sort by (-len, text), keep first max_n.
    """
    uniq = list(dict.fromkeys(words))
    if max_n <= 0:
        return []
    if len(uniq) <= max_n:
        return uniq
    ranked = sorted(uniq, key=lambda w: (-len(w), w))
    return ranked[:max_n]


def _reference_onnx_paths(
    ref_root: Path,
    encoder_onnx: str | None,
    ctc_onnx: str | None,
    tokens_path: str | None,
) -> Tuple[Path, Path, Path]:
    model_dir = ref_root / "model"
    enc_path = Path(encoder_onnx) if encoder_onnx else _pick_onnx(model_dir, ["Fun-ASR-Nano-Encoder-Adaptor"])
    ctc_path = Path(ctc_onnx) if ctc_onnx else _pick_onnx(model_dir, ["Fun-ASR-Nano-CTC"])
    tok_path = Path(tokens_path) if tokens_path else model_dir / "tokens.txt"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokens.txt not found: {tok_path}")
    return enc_path, ctc_path, tok_path


def _reference_build_ort_sessions(
    enc_path: Path,
    ctc_path: Path,
    tok_path: Path,
    onnx_provider: str,
    hots_lines: List[str],
) -> Tuple[Any, Any]:
    """Create Encoder + CTCDecoder ORT sessions (includes warmup inside constructors)."""
    from fun_asr_gguf.inference.ctc_decoder import CTCDecoder
    from fun_asr_gguf.inference.encoder import AudioEncoder

    encoder = AudioEncoder(model_path=str(enc_path), onnx_provider=onnx_provider, dml_pad_to=30)
    ctc_decoder = CTCDecoder(
        model_path=str(ctc_path),
        tokens_path=str(tok_path),
        onnx_provider=onnx_provider,
        dml_pad_to=30,
        hotwords=hots_lines,
        similar_threshold=0.6,
    )
    return encoder, ctc_decoder


def _reference_run_retrieval_inference(
    encoder: Any,
    ctc_decoder: Any,
    audio_path: Path,
    max_hotwords: int,
    ctc_topk: int,
    cap_retrieval: bool,
) -> Tuple[List[str], str, Dict[str, Any]]:
    """load_audio → encode → decode (+ optional cap). Does not create ORT sessions."""
    from fun_asr_gguf.inference.audio import load_audio

    t0 = time.perf_counter()
    audio = load_audio(str(audio_path), sample_rate=16000)
    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio, dtype=np.float32)
    audio = audio.astype(np.float32)
    t_audio_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, enc_out = encoder.encode(audio)
    t_encoder = time.perf_counter() - t0

    t0 = time.perf_counter()
    ctc_results, hotwords, ctc_stage = ctc_decoder.decode(
        enc_out,
        enable_ctc=True,
        max_hotwords=max_hotwords,
        top_k=ctc_topk,
    )
    t_ctc_total = time.perf_counter() - t0
    ctc_text = "".join([r.text for r in ctc_results]) if ctc_results else ""

    hw = list(hotwords)
    t_cap = 0.0
    if cap_retrieval:
        t0 = time.perf_counter()
        hw = _cap_reference_hotword_list(hw, max_hotwords)
        t_cap = time.perf_counter() - t0
    timings = {
        "audio_load": t_audio_load,
        "encoder_total": t_encoder,
        "ctc_total": t_ctc_total,
        "optional_cap": t_cap,
        "ctc_detail": dict(ctc_stage),
    }
    return hw, ctc_text, timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hotword retrieval: current CTC-RAG vs reference CTC pipeline.")
    parser.add_argument("--audio", required=True, type=Path, help="Audio file path.")
    parser.add_argument("--hots", required=True, type=Path, help="Hotword library (one phrase per line).")
    parser.add_argument("--hot_ref", required=True, type=Path, help="Ground-truth hotwords present in the audio.")
    parser.add_argument(
        "--max_hotwords",
        type=int,
        default=32,
        help=(
            "CURRENT CTC-RAG: passed to the CTC-RAG lattice retrieval in model.py. "
            "REFERENCE: passed to PhonemeCorrector.correct(k=...); radar hits are NOT capped by this "
            "(see --ref_retrieval_cap)."
        ),
    )
    parser.add_argument("--current_model", default="FunAudioLLM/Fun-ASR-Nano-2512", help="Model id for FunASRNano.")
    parser.add_argument("--current_hub", default="ms", choices=["ms", "hf"])
    parser.add_argument("--device", default=None, help="cuda:0 / cpu / mps (default: auto).")
    parser.add_argument("--language", default=None, help="Optional ASR language hint for current model.")
    parser.add_argument("--ref_root", type=Path, default=_REPO_ROOT / "refs" / "Fun-ASR-GGUF", help="Fun-ASR-GGUF root.")
    parser.add_argument(
        "--onnx_provider",
        default="AUTO",
        help=(
            "ORT provider for REFERENCE (CPU/CUDA/DML/...). "
            "AUTO: CUDA if --device is cuda:* and GPU available, else CPU. "
            "Timing vs current is misleading if current uses GPU and this stays CPU."
        ),
    )
    parser.add_argument("--ref_encoder_onnx", default=None)
    parser.add_argument("--ref_ctc_onnx", default=None)
    parser.add_argument("--ref_tokens", default=None)
    parser.add_argument(
        "--ctc_topk",
        type=int,
        default=30,
        help=(
            "Reference HotwordRadar: only first K columns of ONNX CTC output are used (slices ids/probs). "
            "Smaller K less CPU in radar; ONNX still runs full exported top-K forward. "
            "Typical export K=30 (CTCHeadExportWrapper); values above K have no effect."
        ),
    )
    parser.add_argument(
        "--ref_retrieval_cap",
        action="store_true",
        help=(
            "After reference decode, truncate the merged candidate list to --max_hotwords "
            "(stable: dedupe, sort by phrase length desc then lex). Improves length parity with "
            "current RAG but ranking differs from current's score-based cap."
        ),
    )
    parser.add_argument("--skip_reference", action="store_true", help="Only run current-repo retrieval.")
    args = parser.parse_args()

    for p in (args.audio, args.hots, args.hot_ref):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    hots_lines = _load_text_lines(args.hots)
    ref_lines = _load_text_lines(args.hot_ref)
    hot_ref_set = set(ref_lines)

    lib_set = set(hots_lines)
    not_in_lib = sorted(hot_ref_set - lib_set)
    if not_in_lib:
        print(
            "WARNING: the following hot_ref entries are not in --hots library (string mismatch); "
            "recall can never be 1.0 unless lines match exactly:"
        )
        for w in not_in_lib:
            print(f"  - {w!r}")

    print(
        "\n[Fairness note]\n"
        f"  max_hotwords={args.max_hotwords}: CURRENT CTC-RAG and REFERENCE both use Top-K CTC lattice retrieval.\n"
        "  REFERENCE pipeline merges radar-detected phrases with ALL phoneme-corrector match/similar "
        "candidates; list length often exceeds max_hotwords unless you pass --ref_retrieval_cap.\n"
        f"  ctc_topk={args.ctc_topk}: used by CURRENT CTC-RAG and REFERENCE (radar over CTC top-K lattice).\n"
        f"  ref_retrieval_cap={'ON' if args.ref_retrieval_cap else 'OFF'} (truncate reference list to max_hotwords).\n"
        "  Timing: compare ORT provider to PyTorch device (AUTO aligns CUDA with cuda:*).\n"
    )

    device = args.device
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            device = "mps"

    from model import FunASRNano

    t0_load = time.perf_counter()
    model, mkwargs = FunASRNano.from_pretrained(model=args.current_model, device=device, hub=args.current_hub)
    model.eval()

    prep_kwargs = dict(mkwargs)
    prep_kwargs["device"] = device
    if args.language is not None:
        prep_kwargs["language"] = args.language

    onnx_provider = _resolve_onnx_provider(args.onnx_provider, device)
    cli_prov = (args.onnx_provider or "AUTO").strip().upper()
    if str(device).lower().startswith("cuda") and onnx_provider == "CPU" and cli_prov == "CPU":
        print(
            "\nWARNING: PyTorch runs on CUDA but you forced reference --onnx_provider CPU — "
            "reference wall time is not comparable to GPU PyTorch. "
            "Omit the flag or use AUTO/CUDA for ORT on GPU.\n"
        )

    t_load_model = time.perf_counter() - t0_load
    print(f"\n[Timing] Model load (from_pretrained): {t_load_model:.3f} s")

    t0 = time.perf_counter()
    cur_retrieved, cur_ctc, cur_stage_timings = _run_current_ctc_rag(
        model,
        args.audio.resolve(),
        args.hots.resolve(),
        args.max_hotwords,
        args.ctc_topk,
        prep_kwargs,
    )
    t_current = time.perf_counter() - t0
    cur_m = _metrics(hot_ref_set, cur_retrieved)
    _print_side(
        "CURRENT (PyTorch CTC-RAG)",
        cur_m,
        cur_ctc,
        elapsed_s=t_current,
        stage_timings=cur_stage_timings,
    )

    if args.skip_reference:
        print("\n(skip_reference) Reference side not run.")
        print(f"\n[Timing] Retrieval only — current ctc_rag: {t_current * 1000.0:.2f} ms")
        return

    ref_root = args.ref_root.resolve()
    _bootstrap_ref_inference(ref_root)
    enc_path, ctc_path, tok_path = _reference_onnx_paths(
        ref_root,
        args.ref_encoder_onnx,
        args.ref_ctc_onnx,
        args.ref_tokens,
    )
    t_ref_onnx0 = time.perf_counter()
    encoder, ctc_decoder = _reference_build_ort_sessions(
        enc_path,
        ctc_path,
        tok_path,
        onnx_provider,
        hots_lines,
    )
    t_ref_onnx_load = time.perf_counter() - t_ref_onnx0
    print(f"\n[Timing] Reference ORT session build (Encoder+CTCDecoder, includes warmup): {t_ref_onnx_load:.3f} s")

    t0 = time.perf_counter()
    ref_retrieved, ref_ctc, ref_stage_timings = _reference_run_retrieval_inference(
        encoder,
        ctc_decoder,
        args.audio.resolve(),
        args.max_hotwords,
        args.ctc_topk,
        args.ref_retrieval_cap,
    )
    t_reference = time.perf_counter() - t0
    ref_m = _metrics(hot_ref_set, ref_retrieved)
    _print_side(
        "REFERENCE (Fun-ASR-GGUF CTCDecoder)",
        ref_m,
        ref_ctc,
        elapsed_s=t_reference,
        elapsed_label="Wall time (audio→encode→decode; ORT session build excluded):",
        stage_timings=ref_stage_timings,
    )

    print(f"\n{'=' * 20} SUMMARY {'=' * 20}")
    print(f"hot_ref count: {len(hot_ref_set)}")
    print(f"Current CTC-RAG    recall: {cur_m['recall']:.4f}  precision_on_ref: {cur_m['precision_on_ref']:.4f}")
    print(f"Reference ORT      recall: {ref_m['recall']:.4f}  precision_on_ref: {ref_m['precision_on_ref']:.4f}")
    hits_cur = set(cur_m["hits"])
    hits_ref = set(ref_m["hits"])
    print(f"Hits by both:           {sorted(hits_cur & hits_ref)}")
    print(f"Hits only current:      {sorted(hits_cur - hits_ref)}")
    print(f"Hits only reference:    {sorted(hits_ref - hits_cur)}")
    print("\n[Timing]")
    print(f"  Model load (from_pretrained): {t_load_model:.3f} s")
    print(f"  PyTorch device (current):     {device}")
    print(f"  ONNXRuntime provider (ref):   {onnx_provider}  (cli --onnx_provider={args.onnx_provider!r})")
    print(f"  Reference ORT session build:    {t_ref_onnx_load:.3f} s  (Encoder+CTCDecoder init+warmup)")
    print(f"  Retrieval path — current ctc_rag:  {t_current * 1000.0:.2f} ms ({t_current:.4f} s)")
    print("    Current ctc_rag stage breakdown:")
    for name, value in cur_stage_timings.items():
        if isinstance(value, dict):
            print(f"      {name}:")
            for sub_name, sub_value in value.items():
                print(f"        {sub_name:<24} {sub_value * 1000.0:9.2f} ms ({sub_value:.4f} s)")
        else:
            print(f"      {name:<26} {value * 1000.0:9.2f} ms ({value:.4f} s)")
    print(
        f"  Retrieval path — reference: {t_reference * 1000.0:.2f} ms ({t_reference:.4f} s)  "
        f"(audio→encode→decode only; excludes ORT load above)"
    )
    print("    Reference stage breakdown:")
    for name, value in ref_stage_timings.items():
        if isinstance(value, dict):
            print(f"      {name}:")
            for sub_name, sub_value in value.items():
                print(f"        {sub_name:<24} {sub_value * 1000.0:9.2f} ms ({sub_value:.4f} s)")
        else:
            print(f"      {name:<26} {value * 1000.0:9.2f} ms ({value:.4f} s)")
    retrieval_paths = {
        "current ctc_rag": t_current,
        "reference ORT": t_reference,
    }
    fastest_name, fastest_time = min(retrieval_paths.items(), key=lambda item: item[1])
    slowest_name, slowest_time = max(retrieval_paths.items(), key=lambda item: item[1])
    if slowest_time > 0 and fastest_time > 0:
        ratio = slowest_time / max(fastest_time, 1e-9)
        print(f"  Fastest side: {fastest_name}  (slowest={slowest_name}, slow/fast ratio ≈ {ratio:.2f}x)")


if __name__ == "__main__":
    main()
