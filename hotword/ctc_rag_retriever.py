# coding: utf-8
"""
Standalone CTC-RAG retriever for Fun-ASR-Nano.

This module keeps `ctc_rag_hotword.py` as the pure retrieval algorithm layer and
adds a reusable wrapper that can:
1. load Fun-ASR-Nano CTC-related components,
2. run audio -> frontend -> audio_encoder -> ctc_decoder -> log_probs,
3. execute CTC-RAG hotword retrieval on top of those log-probs.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from .hot_phoneme import PhonemeCorrector
from .ctc_rag_hotword import run_ctc_rag_hotword_pipeline

logger = logging.getLogger(__name__)

AudioInput = Union[str, np.ndarray, torch.Tensor, Tuple[Union[np.ndarray, torch.Tensor], int]]


@dataclass
class CTCRagResult:
    greedy_text: str
    integrated_text: str
    retrieved_hotwords: List[str]
    radar_hotwords: List[str] = field(default_factory=list)
    extra_hotwords: List[str] = field(default_factory=list)
    hotword_scores: Dict[str, float] = field(default_factory=dict)
    correction_result: Any | None = None
    details: Dict[str, Any] = field(default_factory=dict)


class CTCRagRetriever:
    """Reusable CTC-RAG hotword retriever backed by Fun-ASR-Nano CTC components."""

    def __init__(
        self,
        nano_model: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        nano_remote_code: Optional[str] = None,
        device: Optional[str] = None,
        hub: str = "ms",
        disable_update: bool = True,
        ctc_only: bool = True,
        threshold: float = 0.7,
        similar_threshold: float = 0.6,
        min_hotword_chars: int = 2,
        min_hotword_phonemes: int = 3,
        min_char_coverage: float = 0.6,
    ) -> None:
        self.nano_model = nano_model
        self.nano_remote_code = nano_remote_code or self._default_remote_code()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hub = hub
        self.disable_update = disable_update
        self.ctc_only = ctc_only

        self.threshold = threshold
        self.similar_threshold = similar_threshold
        self.min_hotword_chars = min_hotword_chars
        self.min_hotword_phonemes = min_hotword_phonemes
        self.min_char_coverage = min_char_coverage

        self._hotword_file_lines_cache: Dict[str, Tuple[float, List[str]]] = {}
        self._hotword_corrector_cache: Dict[Tuple[Any, ...], Optional[PhonemeCorrector]] = {}
        self._active_hotword_lines: List[str] = []
        self._active_hotword_source: Optional[str] = None
        self._active_corrector: Optional[PhonemeCorrector] = None

        self._wrapper = None
        self._load_ctc_model()

    @classmethod
    def from_funasr_nano(cls, **kwargs) -> "CTCRagRetriever":
        return cls(**kwargs)

    @staticmethod
    def _default_remote_code() -> Optional[str]:
        repo_model = Path(__file__).resolve().parent.parent / "model.py"
        return str(repo_model) if repo_model.exists() else None

    def _load_ctc_model(self) -> None:
        from funasr import AutoModel as FunASRAutoModel

        load_kwargs: Dict[str, Any] = {
            "model": self.nano_model,
            "device": self.device,
            "hub": self.hub,
            "disable_update": self.disable_update,
        }
        if self.nano_remote_code is not None:
            abs_remote_code = os.path.abspath(self.nano_remote_code)
            load_kwargs["remote_code"] = abs_remote_code
            load_kwargs["trust_remote_code"] = True

            module_name = os.path.splitext(os.path.basename(abs_remote_code))[0]
            sys.modules.pop(module_name, None)
            remote_dir = os.path.dirname(abs_remote_code)
            if remote_dir not in sys.path:
                sys.path.insert(0, remote_dir)

        logger.info("Loading Fun-ASR-Nano CTC retriever from %s", self.nano_model)
        wrapper = FunASRAutoModel(**load_kwargs)
        inner = wrapper.model

        if inner.ctc_decoder is None:
            raise RuntimeError("Loaded model does not provide a CTC decoder.")

        self.audio_encoder = inner.audio_encoder
        self.ctc_decoder = inner.ctc_decoder
        self.ctc = inner.ctc
        self.ctc_tokenizer = inner.ctc_tokenizer
        self.blank_id = inner.blank_id
        self.frontend = wrapper.kwargs.get("frontend")
        if self.frontend is None:
            raise RuntimeError("Failed to extract frontend from AutoModel.")

        self.audio_encoder.eval()
        self.ctc_decoder.eval()
        self.ctc.eval()

        if self.ctc_only:
            if getattr(inner, "llm", None) is not None:
                del inner.llm
                inner.llm = None
            if getattr(inner, "audio_adaptor", None) is not None:
                del inner.audio_adaptor
                inner.audio_adaptor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("CTC-only mode enabled: dropped LLM and audio_adaptor references.")
        else:
            self._wrapper = wrapper

    @staticmethod
    def _normalize_hotword_lines(lines: Iterable[str]) -> List[str]:
        normalized: List[str] = []
        for line in lines:
            item = str(line).strip()
            if not item or item.startswith("#"):
                continue
            normalized.append(item)
        return normalized

    def _load_hotword_lines_from_file(self, path: str) -> List[str]:
        if not path:
            return []
        ap = os.path.abspath(path)
        try:
            mtime = os.path.getmtime(ap)
        except OSError as exc:
            logger.warning("Hotword file is not readable: %s (%s)", ap, exc)
            return []

        cached = self._hotword_file_lines_cache.get(ap)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        with open(ap, "r", encoding="utf-8") as f:
            lines = self._normalize_hotword_lines(f.readlines())
        self._hotword_file_lines_cache[ap] = (mtime, lines)
        return lines

    def _get_hotword_corrector(
        self,
        hotword_lines: List[str],
        source_path: Optional[str] = None,
    ) -> Optional[PhonemeCorrector]:
        if not hotword_lines:
            return None

        if source_path:
            ap = os.path.abspath(source_path)
            try:
                mtime = os.path.getmtime(ap)
            except OSError:
                mtime = None
            cache_key = ("file", ap, mtime)
        else:
            cache_key = ("list", tuple(hotword_lines))

        corrector = self._hotword_corrector_cache.get(cache_key)
        if corrector is None and cache_key not in self._hotword_corrector_cache:
            try:
                corrector = PhonemeCorrector(
                    threshold=self.threshold,
                    similar_threshold=self.similar_threshold,
                    min_hotword_chars=self.min_hotword_chars,
                    min_hotword_phonemes=self.min_hotword_phonemes,
                    min_char_coverage=self.min_char_coverage,
                )
                n = corrector.update_hotwords("\n".join(hotword_lines))
                logger.info("Loaded %s hotwords into PhonemeCorrector", n)
            except Exception as exc:
                logger.warning("Failed to initialize PhonemeCorrector: %s", exc)
                corrector = None
            self._hotword_corrector_cache[cache_key] = corrector
        return self._hotword_corrector_cache[cache_key]

    def load_hotwords(self, hotwords: Union[str, Iterable[str]]) -> int:
        source_path = None
        if isinstance(hotwords, str):
            if os.path.exists(hotwords):
                source_path = hotwords
                hotword_lines = self._load_hotword_lines_from_file(hotwords)
            else:
                hotword_lines = self._normalize_hotword_lines(hotwords.splitlines())
        else:
            hotword_lines = self._normalize_hotword_lines(hotwords)

        self._active_hotword_lines = hotword_lines
        self._active_hotword_source = source_path
        self._active_corrector = self._get_hotword_corrector(hotword_lines, source_path=source_path)
        return len(hotword_lines)

    @staticmethod
    def _as_waveform_tensor(audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            tensor = torch.from_numpy(audio)
        else:
            tensor = audio.detach().cpu()
        tensor = tensor.to(torch.float32)
        if tensor.dim() == 2:
            tensor = tensor.mean(dim=0)
        elif tensor.dim() > 2:
            raise ValueError(f"Unsupported waveform shape: {tuple(tensor.shape)}")
        return tensor.contiguous()

    def _prepare_audio_source(self, audio: AudioInput) -> Union[str, torch.Tensor]:
        if isinstance(audio, str):
            return audio
        if isinstance(audio, tuple) and len(audio) == 2:
            wav, sample_rate = audio
            waveform = self._as_waveform_tensor(wav)
            from funasr.utils.load_utils import load_audio_text_image_video

            return load_audio_text_image_video(
                waveform,
                fs=self.frontend.fs,
                audio_fs=int(sample_rate),
                data_type="sound",
            )
        if isinstance(audio, (np.ndarray, torch.Tensor)):
            return self._as_waveform_tensor(audio)
        raise ValueError(f"Unsupported audio input type: {type(audio)}")

    def encode_audio(self, audio: AudioInput) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

        t0 = time.perf_counter()
        prepared_audio = self._prepare_audio_source(audio)
        if isinstance(prepared_audio, str):
            data_src = load_audio_text_image_video(
                prepared_audio,
                fs=self.frontend.fs,
                data_type="sound",
            )
        else:
            data_src = prepared_audio
        t_load = time.perf_counter() - t0

        t0 = time.perf_counter()
        speech, speech_lengths = extract_fbank(
            data_src,
            data_type="sound",
            frontend=self.frontend,
            is_final=True,
        )
        t_feat = time.perf_counter() - t0

        speech = speech.to(self.device)
        speech_lengths = speech_lengths.to(self.device)

        t0 = time.perf_counter()
        if speech_lengths.dim() > 1:
            speech_lengths = speech_lengths[:, 0]
        encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)
        t_encode = time.perf_counter() - t0

        return encoder_out, encoder_out_lens, {
            "load_audio": t_load,
            "extract_feat": t_feat,
            "audio_encoder": t_encode,
        }

    @torch.no_grad()
    def compute_log_probs(self, audio: AudioInput) -> Tuple[torch.Tensor, Dict[str, Any]]:
        encoder_out, encoder_out_lens, timings = self.encode_audio(audio)

        t0 = time.perf_counter()
        decoder_out, decoder_out_lens = self.ctc_decoder(encoder_out, encoder_out_lens)
        t_decoder = time.perf_counter() - t0

        t0 = time.perf_counter()
        ctc_logits = self.ctc.log_softmax(decoder_out)
        utterance_len = int(encoder_out_lens[0].item())
        log_probs = ctc_logits[0, :utterance_len, :]
        t_log_softmax = time.perf_counter() - t0

        timings.update(
            {
                "ctc_decoder": t_decoder,
                "ctc_log_softmax": t_log_softmax,
            }
        )
        meta = {
            "encoder_out": encoder_out,
            "encoder_out_lens": encoder_out_lens,
            "decoder_out_lens": decoder_out_lens,
            "timings": timings,
        }
        return log_probs, meta

    def retrieve_from_log_probs(
        self,
        log_probs: torch.Tensor,
        max_hotwords: int = 32,
        ctc_topk: int = 30,
        hotwords: Optional[Union[str, Iterable[str]]] = None,
    ) -> CTCRagResult:
        if hotwords is not None:
            self.load_hotwords(hotwords)
        if not self._active_hotword_lines:
            raise ValueError("No hotwords loaded. Call load_hotwords() first or pass hotwords=.")

        integrated_text, retrieved_hotwords, meta = run_ctc_rag_hotword_pipeline(
            log_probs=log_probs,
            blank_id=self.blank_id,
            hotword_lines=self._active_hotword_lines,
            ctc_tokenizer=self.ctc_tokenizer,
            corrector=self._active_corrector,
            max_hotwords=max_hotwords,
            ctc_topk=ctc_topk,
        )

        score_map: Dict[str, float] = {hw: 0.0 for hw in meta.get("radar_texts", [])}
        correction = meta.get("correction")
        if correction is not None:
            for _, hw, score in list(correction.matchs) + list(correction.similars):
                score_map[hw] = max(score_map.get(hw, 0.0), float(score))

        return CTCRagResult(
            greedy_text=meta.get("greedy_text", ""),
            integrated_text=integrated_text,
            retrieved_hotwords=retrieved_hotwords,
            radar_hotwords=meta.get("radar_texts", []),
            extra_hotwords=meta.get("extra_hotwords", []),
            hotword_scores=score_map,
            correction_result=correction,
            details=meta,
        )

    @torch.no_grad()
    def retrieve(
        self,
        audio: AudioInput,
        max_hotwords: int = 32,
        ctc_topk: int = 30,
        hotwords: Optional[Union[str, Iterable[str]]] = None,
    ) -> CTCRagResult:
        log_probs, meta = self.compute_log_probs(audio)
        result = self.retrieve_from_log_probs(
            log_probs,
            max_hotwords=max_hotwords,
            ctc_topk=ctc_topk,
            hotwords=hotwords,
        )
        result.details = {
            **result.details,
            "audio_timings": meta.get("timings", {}),
        }
        return result
