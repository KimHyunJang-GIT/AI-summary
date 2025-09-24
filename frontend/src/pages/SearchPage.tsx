import React, { useState } from "react";
import { ResultCard } from "../components/ResultCard";
import {
  search,
  sendFeedback,
  resetSession,
  triggerReindex,
  type SearchHit,
  type SessionSummary,
  type FeedbackAction,
  type ChatMessage,
  type AnswerSource,
} from "../api/client";

const UI_EVENT_KEY = "infopilot_ui_events";
const MAX_UI_EVENT_LOGS = 50;

const recordUiEvent = (event: Record<string, unknown>) => {
  try {
    const raw = localStorage.getItem(UI_EVENT_KEY);
    const parsed: unknown[] = raw ? JSON.parse(raw) : [];
    if (Array.isArray(parsed)) {
      const next = [...parsed, { ...event, ts: new Date().toISOString() }];
      if (next.length > MAX_UI_EVENT_LOGS) {
        next.splice(0, next.length - MAX_UI_EVENT_LOGS);
      }
      localStorage.setItem(UI_EVENT_KEY, JSON.stringify(next));
    } else {
      localStorage.setItem(
        UI_EVENT_KEY,
        JSON.stringify([{ ...event, ts: new Date().toISOString() }])
      );
    }
  } catch (err) {
    console.warn("failed to persist UI event", err);
  }
};

export const SearchPage: React.FC = () => {
  const [query, setQuery] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [recentQueries, setRecentQueries] = useState<string[]>([]);
  const [preferredExts, setPreferredExts] = useState<string[]>([]);
  const [ownerPrior, setOwnerPrior] = useState<string[]>([]);
  const [results, setResults] = useState<SearchHit[]>([]);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [answerSource, setAnswerSource] = useState<AnswerSource>("none");
  const [llmError, setLlmError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await search({ query, session_id: sessionId ?? undefined });
      setResults(res.results ?? []);
      if (res.session_id ?? null) {
        setSessionId(res.session_id ?? null);
      }
      setChatHistory(res.history ?? []);
      setAnswerSource(res.answer_source ?? "none");
      setLlmError(res.llm_error ?? null);
      const summary: SessionSummary | undefined = res.session;
      if (summary) {
        setRecentQueries(summary.recent_queries ?? []);
        setPreferredExts(summary.preferred_exts ?? []);
        setOwnerPrior(summary.owner_prior ?? []);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const send = async (hit: SearchHit, action: FeedbackAction) => {
    try {
      const res = await sendFeedback({
        session_id: sessionId ?? undefined,
        doc_id: (hit as any).doc_id,
        path: hit.path,
        ext: hit.ext,
        owner: (hit as any).owner,
        action,
      });
      if (res.session_id ?? null) {
        setSessionId(res.session_id ?? null);
      }
      const summary = res.session;
      if (summary) {
        setRecentQueries(summary.recent_queries ?? []);
        setPreferredExts(summary.preferred_exts ?? []);
        setOwnerPrior(summary.owner_prior ?? []);
      }
      recordUiEvent({ type: "feedback", action, path: hit.path, status: "ok" });
    } catch (err) {
      console.warn("feedback failed", err);
      recordUiEvent({ type: "feedback", action, path: hit.path, status: "error", error: String(err) });
    }
  };

  const handleOpen = (hit: SearchHit) => send(hit, "click");
  const handleOpenFolder = (hit: SearchHit) => send(hit, "click");
  const handlePin = (hit: SearchHit) => send(hit, "pin");
  const handleLike = (hit: SearchHit) => send(hit, "like");
  const handleDislike = (hit: SearchHit) => send(hit, "dislike");

  const handleReset = async () => {
    const res = await resetSession(sessionId ?? undefined);
    setSessionId(res.session_id ?? null);
    setRecentQueries(res.recent_queries ?? []);
    setPreferredExts([]);
    setOwnerPrior([]);
    setResults([]);
    setChatHistory(res.history ?? []);
    setAnswerSource("none");
    setLlmError(null);
  };

  const handleReindex = async () => {
    await triggerReindex(false);
  };

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-4">
      <header className="space-y-2">
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 border rounded px-3 py-2"
            placeholder="검색어를 입력하세요"
          />
          <button className="btn" onClick={handleSearch} disabled={loading}>
            {loading ? "검색 중…" : "검색"}
          </button>
        </div>
        <div className="flex gap-2 text-sm">
          <button className="btn-secondary" onClick={handleReset}>
            세션 초기화
          </button>
          <button className="btn-secondary" onClick={handleReindex}>
            재색인 요청
          </button>
        </div>
        {sessionId && (
          <div className="text-xs text-gray-500">세션 ID: {sessionId}</div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-gray-500">
          {recentQueries.length > 0 && (
            <div>
              <div className="font-semibold">최근 질의</div>
              <ul className="space-y-1">
                {recentQueries.slice(-5).map((item, idx) => (
                  <li key={idx} className="truncate" title={item}>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {(preferredExts.length > 0 || ownerPrior.length > 0) && (
            <div className="space-y-1">
              {preferredExts.length > 0 && (
                <div>
                  <div className="font-semibold">선호 확장자</div>
                  <div className="flex flex-wrap gap-1">
                    {preferredExts.map((ext) => (
                      <span key={ext} className="badge">
                        {ext}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {ownerPrior.length > 0 && (
                <div>
                  <div className="font-semibold">선호 작성자</div>
                  <div className="flex flex-wrap gap-1">
                    {ownerPrior.map((owner) => (
                      <span key={owner} className="badge">
                        {owner}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
        {error && <div className="text-red-600 text-sm">{error}</div>}
      </header>
      <main className="grid gap-3">
        {chatHistory.length > 0 && (
          <section className="space-y-2 bg-white border rounded-lg shadow-sm p-4">
            <div className="text-sm font-semibold text-gray-600">대화 기록</div>
            <div className="space-y-3">
              {chatHistory.map((msg, idx) => {
                const isAssistant = msg.role === "assistant";
                return (
                  <div
                    key={`${msg.role}-${idx}`}
                    className={`flex ${isAssistant ? "justify-start" : "justify-end"}`}
                  >
                    <div
                      className={`max-w-3xl whitespace-pre-wrap rounded-lg px-3 py-2 text-sm leading-relaxed ${
                        isAssistant
                          ? "bg-indigo-50 border border-indigo-100 text-indigo-900"
                          : "bg-gray-100 border border-gray-200 text-gray-800"
                      }`}
                    >
                      {msg.text}
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span>응답 모드:</span>
              <span className="font-medium">
                {answerSource === "llm"
                  ? "LLM 생성"
                  : answerSource === "fallback"
                  ? "검색 기반 기본 응답"
                  : "응답 없음"}
              </span>
            </div>
            {llmError && (
              <div className="text-xs text-red-500">
                LLM 호출에 실패하여 기본 응답을 표시했습니다. (원인: {llmError})
              </div>
            )}
          </section>
        )}
        {results.map((hit, idx) => (
          <ResultCard
            key={`${hit.path}-${idx}`}
            hit={hit}
            index={idx}
            onOpen={handleOpen}
            onOpenFolder={handleOpenFolder}
            onPin={handlePin}
            onLike={handleLike}
            onDislike={handleDislike}
          />
        ))}
        {!loading && results.length === 0 && (
          <div className="text-center text-gray-400 text-sm">결과가 없습니다.</div>
        )}
      </main>
    </div>
  );
};
