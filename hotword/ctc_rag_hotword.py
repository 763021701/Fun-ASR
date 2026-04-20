# coding: utf-8
"""
CTC-RAG hotword path on PyTorch CTC log-probs:
top-K lattice -> HotwordRadar -> greedy token stream -> ResultIntegrator -> PhonemeCorrector.

Use with model.inference(..., hotword_mode="ctc_rag", ctc_topk=30, ...).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .radar_scan import HotwordRadar
from .result_integrator import ResultIntegrator


@dataclass
class Token:
    text: str
    timestamp: float
    is_hotword: bool = False


class _PieceTokenizerAdapter:
    """Maps CTC id <-> display piece for HotwordRadar."""

    def __init__(self, ctc_tokenizer: Any, vocab_size: int):
        self._ctc = ctc_tokenizer
        self._n = int(vocab_size)
        self._piece_cache: Dict[int, str] = {}

    def get_piece_size(self) -> int:
        return self._n

    def id_to_piece(self, i: int) -> str:
        i = int(i)
        if i not in self._piece_cache:
            self._piece_cache[i] = self._ctc.decode([i])
        return self._piece_cache[i]


def _merge_hotword_lists_stable(primary: List[str], secondary: List[str]) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for seq in (primary, secondary):
        for item in seq:
            if item and item not in seen:
                seen.add(item)
                merged.append(item)
    return merged


def decode_ctc_indices(
    indices: np.ndarray,
    id2token: Dict[int, str],
    blank_id: int,
) -> Tuple[str, List[Token], Dict[str, float]]:
    """Greedy CTC decode from per-frame token ids."""
    t0 = time.perf_counter()
    frame_shift_ms = 60

    collapsed: List[Tuple[int, int]] = []
    if len(indices) > 0:
        current_id = int(indices[0])
        start_idx = 0
        for i in range(1, len(indices)):
            if int(indices[i]) != current_id:
                collapsed.append((current_id, start_idx))
                current_id = int(indices[i])
                start_idx = i
        collapsed.append((current_id, start_idx))

    results: List[Token] = []
    for token_id, start in collapsed:
        if token_id == blank_id:
            continue
        token_text = id2token.get(token_id, "")
        if not token_text:
            continue
        t_timestamp = max((start * frame_shift_ms) / 1000.0, 0.0)
        results.append(Token(text=token_text, timestamp=t_timestamp))

    full_text = "".join([r.text for r in results])
    t_loop = time.perf_counter() - t0
    timings = {"cast": 0.0, "argmax": 0.0, "loop": t_loop}
    return full_text, results, timings


def run_ctc_rag_hotword_pipeline(
    log_probs: torch.Tensor,
    blank_id: int,
    hotword_lines: List[str],
    ctc_tokenizer: Any,
    corrector: Any,
    max_hotwords: int,
    ctc_topk: int,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Args:
        log_probs: [T, V] tensor (log-softmax), one utterance.
    Returns:
        integrated_text: CTC string after radar+integrate.
        hotword_list: radar texts ∪ phoneme-corrector extras.
        meta: small debug dict.
    """
    if log_probs.dim() != 2:
        raise ValueError(f"expected log_probs [T,V], got shape {tuple(log_probs.shape)}")
    device = log_probs.device
    V = int(log_probs.shape[-1])
    K = min(int(ctc_topk), V)
    t0 = time.perf_counter()
    vals, idx = torch.topk(log_probs, k=K, dim=-1)
    indices_2d = idx.detach().cpu().numpy().astype(np.int64, copy=False)
    topk_log_probs = vals.detach().cpu().numpy().astype(np.float64, copy=False)
    topk_probs = np.exp(topk_log_probs)
    t_topk = time.perf_counter() - t0

    t0 = time.perf_counter()
    adapter = _PieceTokenizerAdapter(ctc_tokenizer, V)
    radar = HotwordRadar(hotword_lines, adapter)
    detected_hotwords = radar.scan(indices_2d, topk_probs, top_k=K, blank_id=int(blank_id))
    t_radar = time.perf_counter() - t0

    t0 = time.perf_counter()
    id2token = {i: ctc_tokenizer.decode([i]) for i in range(V)}
    top1_indices = indices_2d[:, 0]
    ctc_text, ctc_results, _ = decode_ctc_indices(top1_indices, id2token, int(blank_id))
    t_greedy = time.perf_counter() - t0

    t0 = time.perf_counter()
    greedy_fmt = [{"text": r.text, "timestamp": r.timestamp} for r in ctc_results]
    if detected_hotwords and greedy_fmt:
        integrated_list = ResultIntegrator.integrate(greedy_fmt, detected_hotwords)
        new_text = "".join([r["text"] for r in integrated_list])
    else:
        integrated_list = greedy_fmt
        new_text = ctc_text
    t_integrate = time.perf_counter() - t0

    radar_texts = [h["text"] for h in detected_hotwords]
    res = None
    extra: List[str] = []
    t_correct = 0.0
    if corrector is not None and getattr(corrector, "hotwords", None) and new_text:
        t0 = time.perf_counter()
        res = corrector.correct(new_text, k=max_hotwords)
        t_correct = time.perf_counter() - t0
        cand: set[str] = set()
        for _, hw, _ in res.matchs:
            cand.add(hw)
        for _, hw, _ in res.similars:
            cand.add(hw)
        extra = list(cand)

    hotwords_out = _merge_hotword_lists_stable(radar_texts, extra)
    meta = {
        "ctc_topk_used": K,
        "radar_hits": len(detected_hotwords),
        "device": str(device),
        "greedy_text": ctc_text,
        "integrated_text": new_text,
        "radar_texts": radar_texts,
        "extra_hotwords": extra,
        "integrated_tokens": integrated_list,
        "correction": res,
        "timings": {
            "topk": t_topk,
            "radar": t_radar,
            "greedy_decode": t_greedy,
            "integrate": t_integrate,
            "hotword_correct": t_correct,
        },
    }
    return new_text, hotwords_out, meta
