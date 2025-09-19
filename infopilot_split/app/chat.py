"""default module split from lnp_chat (auto-split from originals)."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

from ..core.retrieval import Retriever
from ..core.file_finder import StartupSpinner as Spinner


@dataclass
class ChatTurn:
    role: str
    text: str
    hits: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LNPChat:
    corpus_path: Path
    cache_dir: Path = Path("./index_cache")
    topk: int = 10
    similarity_threshold: float = 0.05 # 유사도 임계값 설정

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
            # 1. Retriever는 일단 가능한 많은 후보를 가져옴 (topk * 2, 최소 20개)
            candidate_hits = self.retr.search(query, top_k=max(k * 2, 20))
        finally:
            spin.stop()
        dt = time.time() - t0

        # 2. 유사도 임계값(0.05)을 기준으로 필터링
        filtered_hits = [h for h in candidate_hits if h['similarity'] >= self.similarity_threshold]
        
        # 3. 최종 결과는 필터링된 것에서 topk 만큼만 잘라서 사용
        final_hits = filtered_hits[:k]

        self.history.append(ChatTurn(role="user", text=query))
        self.history.append(ChatTurn(role="assistant", text="", hits=final_hits))

        # 4. 필터링된 결과(final_hits)를 기반으로 답변 생성
        if not final_hits:
            # 기준을 통과한 문서가 하나도 없을 경우, 요청하신 메시지 표시
            answer_lines = [f"‘{query}’와 관련된 내용을 찾지 못했습니다."]
        else:
            answer_lines = [f"‘{query}’와(과) 의미상 유사한 문서 Top {len(final_hits)} (검색 {dt:.2f}s):"]
            for i, h in enumerate(final_hits, 1):
                sim = f"{h['similarity']:.3f}"
                answer_lines.append(f"{i}. {h['path']}  (유사도: {sim})")
                if h.get("summary"):
                    answer_lines.append(f"   요약: {h['summary']}")

        return {
            "answer": "\n".join(answer_lines),
            "hits": final_hits,
            "suggestions": self._suggest_followups(query, final_hits),
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
