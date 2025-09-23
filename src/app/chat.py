from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

from src.config import DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD
from src.core.retrieval import Retriever
from src.core.utils import StartupSpinner as Spinner # 경로 수정

@dataclass
class ChatTurn:
    role: str
    text: str
    hits: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LNPChat:
    corpus_path: Path
    cache_dir: Path
    topk: int = DEFAULT_TOP_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD

    retr: Optional[Retriever] = field(init=False, default=None)
    history: List[ChatTurn] = field(init=False, default_factory=list)
    ready_done: bool = field(init=False, default=False)

    def ready(self, rebuild: bool = False):
        spin = Spinner(prefix="엔진 초기화")
        spin.start()
        try:
            self.retr = Retriever(corpus_path=self.corpus_path, cache_dir=self.cache_dir)
            self.retr.ready(rebuild=rebuild)
            self.ready_done = True
        finally:
            spin.stop()
        print("✅ LNP Chat 준비 완료")

    def ask(self, query: str, topk: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.ready_done: self.ready(rebuild=False)
        k = topk or self.topk
        spin = Spinner(prefix="검색 중")
        spin.start()
        t0 = time.time()
        try:
            candidate_hits = self.retr.search(query, top_k=max(k * 2, 20), filters=filters)
        finally:
            spin.stop()
        dt = time.time() - t0
        filtered_hits = [h for h in candidate_hits if h['similarity'] >= self.similarity_threshold]
        final_hits = filtered_hits[:k]
        self.history.append(ChatTurn(role="user", text=query))
        self.history.append(ChatTurn(role="assistant", text="", hits=final_hits))
        if not final_hits:
            answer_lines = [f"‘{query}’와 관련된 내용을 찾지 못했습니다."]
        else:
            answer_lines = [f"‘{query}’와(과) 의미상 유사한 문서 Top {len(final_hits)} (검색 {dt:.2f}s):"]
            for i, h in enumerate(final_hits, 1):
                sim = f"{h['similarity']:.3f}"
                answer_lines.append(f"{i}. {h['path']}  (유사도: {sim})")
                if h.get("summary"): answer_lines.append(f"   요약: {h['summary']}")
        return {"answer": "\n".join(answer_lines), "hits": final_hits, "suggestions": self._suggest_followups(query, final_hits)}

    def _suggest_followups(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        base = ["이 문서의 핵심 내용을 요약해줘", "위 문서들과 비슷한 다른 문서를 더 찾아줘", "결과를 표 형식으로 정리해줘"] if hits else ["다른 표현으로 같은 의미의 질의를 시도", "문서 유형(엑셀/한글/PDF 등)을 지정해서 검색"]
        seen, out = set(), []
        for s in base:
            if s not in seen: out.append(s); seen.add(s)
        return out[:3]
