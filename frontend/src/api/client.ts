export interface SearchRequest {
  query: string;
  top_k?: number;
  session_id?: string | null;
}

export type FeedbackAction = "click" | "pin" | "like" | "dislike";

export interface FeedbackRequest {
  session_id?: string | null;
  doc_id?: number;
  path?: string;
  ext?: string;
  owner?: string;
  action: FeedbackAction;
}

export interface SearchHit {
  path: string;
  ext: string;
  vector_similarity?: number;
  combined_score?: number;
  preview?: string;
  match_reasons?: string[];
  metadata_matches?: string[];
  score_breakdown?: Record<string, number>;
  session_ext_bonus?: number;
  session_owner_bonus?: number;
}

export interface SessionSummary {
  recent_queries: string[];
  preferred_exts: string[];
  owner_prior: string[];
}

export type AnswerSource = "llm" | "fallback" | "none";

export interface ChatMessage {
  role: "user" | "assistant";
  text: string;
}

export interface SearchResponse {
  session_id?: string | null;
  results: SearchHit[];
  explain: string[][];
  session: SessionSummary;
  answer?: string | null;
  answer_source: AnswerSource;
  history: ChatMessage[];
  llm_error?: string | null;
}

export interface FeedbackResponse {
  session_id?: string | null;
  status: string;
  session?: SessionSummary;
}

export interface SessionResetResponse {
  session_id?: string | null;
  recent_queries: string[];
  history: ChatMessage[];
}

export interface ReindexResponse {
  status: string;
}

const API_BASE = "/api";
const JSON_HEADERS = { "Content-Type": "application/json" } as const;

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload ?? {}),
  });
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export function search(req: SearchRequest): Promise<SearchResponse> {
  return postJson<SearchResponse>("/search", req);
}

export function sendFeedback(req: FeedbackRequest): Promise<FeedbackResponse> {
  return postJson<FeedbackResponse>("/feedback", req);
}

export function resetSession(sessionId?: string | null): Promise<SessionResetResponse> {
  return postJson<SessionResetResponse>("/session/reset", { session_id: sessionId });
}

export function triggerReindex(force = false): Promise<ReindexResponse> {
  return postJson<ReindexResponse>("/reindex", { force });
}
