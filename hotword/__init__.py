# coding: utf-8
"""
热词模块 (从 Fun-ASR-GGUF 项目移植)

提供基于音素编辑距离的热词检索和纠错功能：
- PhonemeCorrector: 基于音素的纠错器 (FastRAG 粗筛 + 精确匹配)
"""

import logging

logger = logging.getLogger("hotword")

from .hot_phoneme import PhonemeCorrector, CorrectionResult
from .ctc_rag_retriever import CTCRagRetriever, CTCRagResult

__all__ = [
    'PhonemeCorrector',
    'CorrectionResult',
    'CTCRagRetriever',
    'CTCRagResult',
    'logger',
]
