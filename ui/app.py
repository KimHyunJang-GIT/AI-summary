import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pathlib import Path
import pandas as pd
import time, joblib

from infopilot_split.app.paths import Paths, have_all_artifacts
from infopilot_split.core.file_finder import FileFinder
from infopilot_split.core.corpus import CorpusBuilder
from infopilot_split.core.indexing import run_indexing
from infopilot_split.app.chat import LNPChat

st.set_page_config(page_title="InfoPilot Split Bundle", page_icon="🧭", layout="wide")
p = Paths().ensure()

st.title("InfoPilot (Split Bundle)")

def show_paths():
    st.caption("경로 및 상태")
    st.code(f"""
base:            {p.base}
data_dir:        {p.data_dir}
models_dir:      {p.models_dir}
cache_dir:       {p.cache_dir}
corpus.csv:      {p.corpus_csv.exists()}
corpus.parquet:  {p.corpus_parquet.exists()}
topic_model:     {p.topic_model.exists()}
""", language="bash")

if "mode" not in st.session_state:
    st.session_state["mode"] = "home"

def go(m): st.session_state["mode"] = m

def home():
    st.subheader("시작하기")
    if have_all_artifacts(p):
        st.success("필요 데이터가 모두 준비되었습니다.")
        c1, c2 = st.columns(2)
        if c1.button("🔁 다시 교육하기"):
            go("train")
        if c2.button("💬 채팅창으로 가기"):
            go("chat")
    else:
        st.warning("학습 데이터가 없습니다. 먼저 '교육시키기'를 실행하세요.")
        if st.button("🚀 교육시키기"):
            go("train")
    st.divider()
    show_paths()

def train():
    st.subheader("교육(스캔 → 코퍼스 → 인덱스)")
    exts = st.text_input("확장자 필터(쉼표 구분)", ".hwp,.doc,.docx,.xlsx,.xlsm,.xls,.pdf,.ppt,.pptx,.csv,.txt")
    do_scan = st.checkbox("드라이브 스캔 실행", value=True)
    found_csv = p.data_dir / "found_files.csv"

    if st.button("▶️ 교육 시작"):
        rows = None
        if do_scan:
            with st.status("드라이브 스캔 중...", expanded=True) as status:
                finder = FileFinder(
                    exts=[e.strip() for e in exts.split(",") if e.strip()],
                    scan_all_drives=True,
                    start_from_current_drive_only=False,
                    follow_symlinks=False,
                    max_depth=None,
                    show_progress=False,
                    estimate_total_dirs=False,
                    startup_banner=False,
                )
                rows = finder.find(run_async=False)
                # Save simple CSV via pandas
                df_scan = pd.DataFrame(rows)
                df_scan.to_csv(found_csv, index=False, encoding="utf-8")
                status.update(label="스캔 완료", state="complete")

        with st.status("텍스트 추출 및 코퍼스 생성...", expanded=True) as status:
            if (p.corpus_parquet.exists()):
                try: p.corpus_parquet.unlink()
                except: pass
            cb = CorpusBuilder(progress=True)
            if rows is None and found_csv.exists():
                df_scan = pd.read_csv(found_csv)
                rows = df_scan.to_dict("records")
            df_corpus = cb.build(rows)  # relies on your original API
            cb.save(df_corpus, p.corpus_parquet)
            # also save success/failure CSVs
            try:
                (df_corpus[df_corpus.get("ok", True)==True]).to_csv(p.data_dir / "corpus_success.csv", index=False, encoding="utf-8")
                (df_corpus[df_corpus.get("ok", False)==False]).to_csv(p.data_dir / "corpus_failure.csv", index=False, encoding="utf-8")
            except Exception as e:
                pass
            status.update(label="코퍼스 생성 완료", state="complete")

        with st.status("벡터 인덱싱 중...", expanded=True) as status:
            run_indexing(corpus_path=p.corpus_parquet, cache_dir=p.cache_dir)
            status.update(label="인덱싱 완료", state="complete")

        with st.status("토픽 모델 메타 저장...", expanded=True) as status:
            meta = {
                "model_name": "jhgan/ko-sroberta-multitask",
                "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "corpus_path": str(p.corpus_parquet),
                "cache_dir": str(p.cache_dir),
            }
            joblib.dump(meta, p.topic_model)
            status.update(label="완료", state="complete")

        st.success("학습이 완료되었습니다. 이제 채팅을 시작할 수 있어요!")
        st.button("💬 채팅창으로 가기", on_click=lambda: go("chat"))

    if found_csv.exists():
        st.write("최근 스캔 결과(일부):")
        try:
            st.dataframe(pd.read_csv(found_csv).head(200))
        except Exception as e:
            st.info(f"스캔 결과 미리보기 실패: {e}")

    st.button("🏠 처음으로", on_click=lambda: go("home"))
    st.divider()
    show_paths()

def chat():
    st.subheader("의미 기반 검색 채팅")
    if not have_all_artifacts(p):
        st.warning("먼저 교육을 실행해주세요.")
        if st.button("🚀 교육시키기"):
            go("train")
        return

    if "chat_engine" not in st.session_state:
        st.session_state["chat_engine"] = LNPChat(corpus_path=p.corpus_parquet, cache_dir=p.cache_dir, topk=10)
        st.session_state["chat_engine"].ready()

    query = st.text_input("질문을 입력하세요", placeholder="예: 세금 관련 파일 찾아줘")
    if st.button("검색") and query.strip():
        with st.status("검색 중...", expanded=False):
            result = st.session_state["chat_engine"].ask(query.strip())
        st.write(result.get("answer", ""))
        hits = result.get("hits", [])
        if hits:
            df = pd.DataFrame([
                {"유사도": h.get("similarity"), "경로": h.get("path"), "요약": h.get("summary", "")}
                for h in hits
            ])
            st.dataframe(df)
        if result.get("suggestions"):
            st.caption("💡 제안")
            for s in result["suggestions"]:
                st.write("- ", s)

    st.button("🏠 처음으로", on_click=lambda: go("home"))
    st.divider()
    show_paths()

mode = st.session_state["mode"]
if mode == "home":
    home()
elif mode == "train":
    train()
elif mode == "chat":
    chat()
else:
    home()
