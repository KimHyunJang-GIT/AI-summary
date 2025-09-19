# -*- coding: utf-8 -*-
"""
LNP Chat: 자연어 대화로 문서 검색/추천
- Retriever(모델/코퍼스/인덱스)를 사용해 사용자 질의 → 유사 문서 Top-K
- 간단한 대화 히스토리, 진행 스피너, 후속질문 제안 포함
"""
from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from retriever import Retriever  # Step3 검색기 재사용

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

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
    model_path: Path
    corpus_path: Path
    cache_dir: Path = Path("./index_cache")
    topk: int = 5
    translate: bool = True # 번역 기능 활성화 옵션

    retr: Optional[Retriever] = field(init=False, default=None)
    translator: Optional[Any] = field(init=False, default=None)
    history: List[ChatTurn] = field(init=False, default_factory=list)
    ready_done: bool = field(init=False, default=False)

    # 초기화: Retriever 및 번역기 준비
    def ready(self, rebuild: bool = False):
        spin = Spinner(prefix="인덱스 로딩/점검")
        spin.start()
        try:
            self.retr = Retriever(
                model_path=self.model_path,
                corpus_path=self.corpus_path,
                cache_dir=self.cache_dir,
            )
            self.retr.ready(rebuild=rebuild)  # 인덱스 없으면 빌드, 있으면 로드
            if self.translate:
                if GoogleTranslator is None:
                    print("\n⚠️ 경고: 'deep-translator' 라이브러리를 찾을 수 없어 번역 기능이 비활성화됩니다.")
                    print("   해결: pip install deep-translator")
                else:
                    try:
                        self.translator = GoogleTranslator(source="auto", target="en")
                    except Exception as exc:
                        print("\n⚠️ 경고: 번역기 초기화에 실패해 번역 기능이 비활성화됩니다.")
                        print(f"   상세: {exc}")
            self.ready_done = True
        finally:
            spin.stop()
        print("✅ LNP Chat 준비 완료 (번역: " + ("활성" if self.translator else "비활성") + ")")

    # 한 턴 처리
    def ask(self, query: str, topk: Optional[int] = None) -> Dict[str, Any]:
        if not self.ready_done:
            self.ready(rebuild=False)
        k = topk or self.topk

        # [번역 기능] 사용자 질문을 영어로 번역
        query_for_search = query
        if self.translator:
            try:
                translated = self.translator.translate(query)
                query_for_search = translated if isinstance(translated, str) else getattr(translated, "text", query)
                print(f"  (질문 번역: '{query}' → '{query_for_search}')")
            except Exception as e:
                print(f"\n[경고] 질문 번역 실패. 원본 질문으로 검색합니다. 오류: {e}")

        # 스피너로 즉시 “살아있음” 표시
        spin = Spinner(prefix="검색 중")
        spin.start()
        t0 = time.time()
        try:
            hits = self.retr.search(query_for_search, top_k=k)
        finally:
            spin.stop()
        dt = time.time() - t0

        # 히스토리 적재 (원본 query 기준)
        self.history.append(ChatTurn(role="user", text=query))

        # 답변 생성(원본 query 기준)
        answer_lines = [f"‘{query}’에 대한 추천 문서 Top {len(hits)} (검색 {dt:.2f}s):"]
        for i, h in enumerate(hits, 1):
            sim = f"{h['similarity']:.2f}"
            answer_lines.append(f"{i}. {h['path']} [{h['ext']}]  유사도={sim}")
            if h.get("preview"):
                label = "미리보기 (원문)" if self.translator else "미리보기"
                answer_lines.append(f"   {label}: {h['preview']}")
        if not hits:
            answer_lines.append("관련 문서를 찾지 못했습니다. 표현을 바꿔보거나 더 구체적으로 적어주세요.")

        answer = "\n".join(answer_lines)
        self.history.append(ChatTurn(role="assistant", text=answer, hits=hits))

        return {
            "answer": answer,
            "hits": hits,
            "suggestions": self._suggest_followups(query, hits),
        }

    # 후속 질문 제안
    def _suggest_followups(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        base = []
        if hits:
            exts = {h["ext"].lower() for h in hits}
            if any(x in exts for x in [".xlsx", ".xls", ".xlsm", ".csv"]):
                base.append("표/컬럼 이름을 기준으로 다시 좁혀줘")
            if any(x in exts for x in [".pdf", ".ppt", ".pptx", ".doc", ".hwp"]):
                base.append("요약/키워드 중심으로 비슷한 문서 더 보여줘")
            base.append("기간(년도/월) 조건을 추가해서 다시 찾아줘")
            base.append("파일명에 포함된 키워드로 재검색")
        else:
            base.append("다른 표현으로 같은 의미의 질의를 시도")
            base.append("문서 유형(엑셀/한글/PDF 등)을 지정해서 검색")
        seen, out = set(), []
        for s in base:
            if s not in seen:
                out.append(s); seen.add(s)
        return out[:3]
