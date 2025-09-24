from __future__ import annotations

import logging
from dataclasses import dataclass
from textwrap import shorten
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


def _load_openai_client(api_key: str, base_url: Optional[str]) -> Any:
    """Lazy-load OpenAI client supporting both v1.x and legacy SDKs."""
    try:
        # Try modern client first
        from openai import OpenAI  # type: ignore

        return OpenAI(api_key=api_key, base_url=base_url)
    except ImportError:
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("openai 패키지가 설치되어 있지 않습니다.") from exc

        openai.api_key = api_key
        if base_url:
            # Newer SDK uses api_base, legacy uses api_base as well.
            setattr(openai, "api_base", base_url)
        return openai


@dataclass
class LLMConfig:
    enabled: bool
    provider: str
    model: Optional[str]
    api_key: Optional[str]
    base_url: Optional[str]
    max_context_docs: int = 4
    temperature: float = 0.2
    language: str = "ko"


class LLMService:
    """Optional LLM orchestrator with graceful fallback responses."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client: Any = None
        self.enabled = bool(
            config.enabled
            and config.api_key
            and config.model
            and config.provider.lower() in {"openai"}
        )

        if not self.enabled:
            logger.info("LLM 서비스 비활성화: provider=%s", config.provider)
            return

        try:
            self.client = _load_openai_client(config.api_key or "", config.base_url)
            logger.info("LLM 서비스 초기화 완료 (provider=%s, model=%s)", config.provider, config.model)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("LLM 클라이언트를 초기화하지 못했습니다: %s", exc)
            self.enabled = False
            self.client = None

    @classmethod
    def from_settings(cls, settings: Any) -> "LLMService":
        provider = getattr(settings, "LLM_PROVIDER", "none") or "none"
        enabled = getattr(settings, "LLM_ENABLED", None)
        if enabled is None:
            enabled = provider.lower() != "none"
        config = LLMConfig(
            enabled=bool(enabled),
            provider=provider,
            model=getattr(settings, "LLM_MODEL", None),
            api_key=getattr(settings, "LLM_API_KEY", None),
            base_url=getattr(settings, "LLM_BASE_URL", None),
            max_context_docs=int(getattr(settings, "LLM_CONTEXT_DOCS", 4) or 4),
            temperature=float(getattr(settings, "LLM_TEMPERATURE", 0.2) or 0.2),
            language=str(getattr(settings, "LLM_LANGUAGE", "ko") or "ko"),
        )
        return cls(config)

    def generate_reply(
        self,
        *,
        user_message: str,
        conversation: Sequence[Tuple[str, str]],
        hits: List[Dict[str, Any]],
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Returns (answer, used_llm, error_message).
        When LLM is disabled or errors, we synthesize a graceful fallback message.
        """

        summary_blocks = self._summarise_hits(hits)

        if not self.enabled or self.client is None:
            return self._fallback_answer(user_message, summary_blocks), False, None

        if not summary_blocks:
            summary_blocks = ["관련 문서를 찾지 못했습니다."]

        messages = self._build_messages(user_message, conversation, summary_blocks)

        try:
            reply = self._invoke_llm(messages)
            if reply:
                return reply, True, None
        except Exception as exc:  # pragma: no cover - depends on runtime
            logger.warning("LLM 응답 생성 실패, 대체 메시지 사용: %s", exc)
            return self._fallback_answer(user_message, summary_blocks), False, str(exc)

        return self._fallback_answer(user_message, summary_blocks), False, None

    # ────────────────────────── 내부 유틸 ──────────────────────────

    def _invoke_llm(self, messages: List[Dict[str, str]]) -> str:
        provider = self.config.provider.lower()
        if provider != "openai":
            raise RuntimeError(f"지원하지 않는 LLM provider: {self.config.provider}")

        client = self.client
        model = self.config.model
        temperature = self.config.temperature

        # OpenAI 1.x client
        chat_api = getattr(client, "chat", None)
        if chat_api and hasattr(chat_api, "completions"):
            completion = chat_api.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            content = completion.choices[0].message.content  # type: ignore[index]
            return content or ""

        # Legacy openai module
        if hasattr(client, "ChatCompletion"):
            completion = client.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            choice = completion.get("choices", [{}])[0]
            message = choice.get("message") or {}
            return message.get("content", "")

        raise RuntimeError("지원하지 않는 OpenAI 클라이언트 타입입니다.")

    def _build_messages(
        self,
        user_message: str,
        conversation: Sequence[Tuple[str, str]],
        summary_blocks: Sequence[str],
    ) -> List[Dict[str, str]]:
        system_prompt = (
            "당신은 기업 내부 문서에 접근할 수 있는 전문 비서입니다. "
            "사용자 요청에 한국어로 친근하지만 간결하게 답변하세요. "
            "제공된 문서 요약을 우선 활용하고, 모르는 내용은 솔직히 모른다고 답하세요."
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for role, text in conversation:
            if role not in {"user", "assistant"}:
                continue
            messages.append({"role": role, "content": text})

        doc_context = "\n\n".join(summary_blocks)
        composed_user = (
            f"사용자 요청:\n{user_message}\n\n"
            f"참고 문서 요약:\n{doc_context}\n"
            "문서에서 찾을 수 없는 내용은 모른다고 답해주세요."
        )
        messages.append({"role": "user", "content": composed_user})
        return messages

    def _summarise_hits(self, hits: Iterable[Dict[str, Any]]) -> List[str]:
        blocks: List[str] = []
        max_docs = max(1, self.config.max_context_docs)
        for idx, hit in enumerate(hits):
            if idx >= max_docs:
                break
            path = str(hit.get("path") or "")
            preview = str(hit.get("preview") or "").strip()
            if preview:
                preview = shorten(preview, width=360, placeholder=" …")
            reasons = hit.get("match_reasons") or []
            reason_text = " · ".join(reasons[:3]) if reasons else ""
            block = f"[{idx + 1}] {path}"
            if reason_text:
                block += f"\n- 근거: {reason_text}"
            if preview:
                block += f"\n- 요약: {preview}"
            blocks.append(block)
        return blocks

    def _fallback_answer(self, user_message: str, summary_blocks: Sequence[str]) -> str:
        if summary_blocks:
            bullet_lines = "\n".join(f"• {block}" for block in summary_blocks)
            return (
                "아래 자료에서 답을 찾았어요. 필요한 부분을 확인해 주세요!\n\n"
                + bullet_lines
            )
        return (
            "관련 문서를 찾지 못했어요. 질문을 조금 더 구체적으로 말씀해 주시면 다시 찾아볼게요."
        )
