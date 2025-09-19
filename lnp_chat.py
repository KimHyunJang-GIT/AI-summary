# -*- coding: utf-8 -*-
"""
LNP Chat: 자연어 대화로 문서 검색/추천
- Retriever(코퍼스/인덱스)를 사용해 사용자 질의 → 유사 문서 Top-K
- 간단한 대화 히스토리, 진행 스피너, 후속질문 제안 포함
"""
from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from retriever import Retriever  # 새로운 의미 기반 검색기

# ──────────────────────────
# 콘솔 스피너 (즉시 피드백)
# ──────────────────────────
class Spinner:
    FRAMES = ["|", "/", "-", "\\"]
    def __init__(self, prefix="검색 준비", interval=0.12):
        self.prefix = prefix
        self.interval = interval
        self._stop = threading.Event()
        self._t = None
        self._i = 0
    def start(self):
        if self._t: return
        def _run():
            while not self._stop.wait(self.interval):
                frame = self.FRAMES[self._i % len(self.FRAMES)]
                self._i += 1
                print(f"\r{self.prefix} {frame} ", end="", flush=True)
        self._t = threading.Thread(target=_run, daemon=True)
        self._t.start()
    def stop(self, clear=True):
        if not self._t: return
        self._stop.set()
        self._t.join()
        if clear:
            print("\r" + " " * 80 + "\r", end="", flush=True)

# ──────────────────────────
# 대화 상태
# ──────────────────────────
@dataclass
class ChatTurn:
    role: str
    text: str
    hits: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LNPChat:
    corpus_path: Path
    cache_dir: Path = Path("./index_cache")
    topk: int = 5

    retr: Optional[Retriever] = field(init=False, default=None)
    history: List[ChatTurn] = field(init=False, default_factory=list)
    ready_done: bool = field(init=False, default=False)

    # 초기화: Retriever 준비(인덱스 로드 or 빌드)
    def ready(self, rebuild: bool = False):
        spin = Spinner(prefix="엔진 초기화")
        spin.start()
        try:
            self.retr = Retriever(
                corpus_path=self.corpus_path,
                cache_dir=self.cache_dir,
            )
            self.retr.ready(rebuild=rebuild)
            self.ready_done = True
        finally:
            spin.stop()
        print("✅ LNP Chat 준비 완료")

    # 한 턴 처리
    def ask(self, query: str, topk: Optional[int] = None) -> Dict[str, Any]:
        if not self.ready_done:
            self.ready(rebuild=False)
        k = topk or self.topk

        spin = Spinner(prefix="검색 중")
        spin.start()
        t0 = time.time()
        try:
            hits = self.retr.search(query, top_k=k)
        finally:
            spin.stop()
        dt = time.time() - t0

        self.history.append(ChatTurn(role="user", text=query))
        self.history.append(ChatTurn(role="assistant", text="", hits=hits))

        answer_lines = [f"‘{query}’와(과) 의미상 유사한 문서 Top {len(hits)} (검색 {dt:.2f}s):"]
        for i, h in enumerate(hits, 1):
            sim = f"{h['similarity']:.3f}"
            answer_lines.append(f"{i}. {h['path']}  (유사도: {sim})")
            if h.get("summary"):
                answer_lines.append(f"   요약: {h['summary']}")
        if not hits:
            answer_lines.append("관련 문서를 찾지 못했습니다. 표현을 바꿔보거나 더 구체적으로 적어주세요.")

        return {
            "answer": "\n".join(answer_lines),
            "hits": hits,
            "suggestions": self._suggest_followups(query, hits),
        }

    # 후속 질문 제안
    def _suggest_followups(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        base = []
        if hits:
            base.append("이 문서의 핵심 내용을 요약해줘")
            base.append("위 문서들과 비슷한 다른 문서를 더 찾아줘")
            base.append("결과를 표 형식으로 정리해줘")
        else:
            base.append("다른 표현으로 같은 의미의 질의를 시도")
            base.append("문서 유형(엑셀/한글/PDF 등)을 지정해서 검색")
        
        seen, out = set(), []
        for s in base:
            if s not in seen:
                out.append(s); seen.add(s)
        return out[:3]
