# -*- coding: utf-8 -*-
"""
pipeline.py (Step2): 텍스트 추출 + 요약(summary/title) 생성 + 코퍼스 저장
- 스캔 결과(파일 목록)를 입력으로 받아 문서별 텍스트 추출
- 추출 텍스트에서 추출식 요약(summary)과 간단 제목(title) 생성
- corpus.(parquet|csv)로 저장 (Parquet 엔진 없으면 CSV 자동 폴백)
"""
from __future__ import annotations
import os, re, sys, time, threading, platform, math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

# ---- 선택 의존성(있으면 사용) ----
try:
    import pandas as pd
except Exception:
    pd = None
try:
    import docx
except Exception:
    docx = None
try:
    import pptx
except Exception:
    pptx = None
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None
try:
    import textract
except Exception:
    textract = None
try:
    import win32com.client
except Exception:
    win32com = None
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# =========================
# 콘솔 진행도 유틸
# =========================
class Spinner:
    FRAMES = ["|", "/", "-", "\\"]
    def __init__(self, prefix="", interval=0.12):
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
                sys.stdout.write(f"\r{self.prefix} {frame} ")
                sys.stdout.flush()
        self._t = threading.Thread(target=_run, daemon=True)
        self._t.start()
    def stop(self, clear=True):
        if not self._t: return
        self._stop.set()
        self._t.join()
        if clear:
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

class ProgressLine:
    """tqdm 미사용 시 한 줄 퍼센트/ETA"""
    def __init__(self, total:int, label:str, update_every:int=10):
        self.total = max(1, total)
        self.label = label
        self.update_every = max(1, update_every)
        self.start = time.time()
        self.n = 0
    def update(self, k:int=1):
        self.n += k
        if (self.n % self.update_every) != 0 and self.n < self.total:
            return
        pct = min(100.0, self.n / self.total * 100.0)
        elapsed = time.time() - self.start
        rate = self.n/elapsed if elapsed>0 else 0
        remain = (self.total - self.n)/rate if rate>0 else 0
        sys.stdout.write(
            f"\r[{pct:5.1f}%] {self.label}  {self.n:,}/{self.total:,}  "
            f"{rate:,.1f}/s  elapsed={self._fmt(elapsed)}  ETA={self._fmt(remain)}   "
        )
        sys.stdout.flush()
    def close(self):
        self.n = self.total
        self.update(0)
        sys.stdout.write("\n"); sys.stdout.flush()
    @staticmethod
    def _fmt(s: float)->str:
        if s==float("inf"): return "∞"
        m, sec = divmod(int(s), 60); h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

# =========================
# 텍스트 클린
# =========================
class TextCleaner:
    _multi = re.compile(r"\s+")
    @classmethod
    def clean(cls, s:str)->str:
        if not s: return ""
        s = "".join(ch if ch.isprintable() or ch in "\t\n\r" else " " for ch in s)
        s = s.replace("\x00"," ")
        return cls._multi.sub(" ", s).strip()

# =========================
# 간단 추출식 요약/제목
# =========================
_SENT_SPLIT = re.compile(r'(?<=[\.!?]|[。！？])\s+|(?<=[다요])\s')
_TOKEN = re.compile(r'[가-힣A-Za-z0-9]{2,}')

def _sent_tokenize(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    out = []
    for s in sents:
        if len(s) > 600:
            out.extend([s[i:i+600] for i in range(0, len(s), 600)])
        else:
            out.append(s)
    return out[:200]

def _tfidf_sentence_scores(sents: List[str]) -> List[Tuple[float,int]]:
    from collections import Counter, defaultdict
    docs = [" ".join(_TOKEN.findall(s.lower())) for s in sents]
    df = defaultdict(int)
    tfs = []
    for d in docs:
        toks = d.split()
        c = Counter(toks)
        tfs.append(c)
        for w in c.keys():
            df[w] += 1
    N = len(docs)
    idf = {w: math.log(1 + N / (1 + dfw)) for w, dfw in df.items()}
    scores = []
    for i, c in enumerate(tfs):
        denom = max(1, sum(c.values()))
        s = sum((tf/denom) * idf.get(w, 0.0) for w, tf in c.items())
        scores.append((s, i))
    return scores

def summarize_extractive(text: str, max_sentences: int = 3) -> str:
    t = (text or "").strip()
    if not t: return ""
    sents = _sent_tokenize(t)
    if not sents: return ""
    scores = _tfidf_sentence_scores(sents)
    scores.sort(reverse=True)
    idx = sorted([i for _, i in scores[:max_sentences]])
    return " ".join([sents[i] for i in idx])

def make_title_like(text: str, fallback: str = "") -> str:
    t = (text or "").strip()
    if not t: return fallback
    first_line = t.splitlines()[0].strip()
    if 3 <= len(first_line) <= 80:
        return first_line
    toks = _TOKEN.findall(t.lower())
    if not toks:
        return fallback or (first_line[:60] if first_line else "")
    from collections import Counter
    top = [w for w,_ in Counter(toks).most_common(6)]
    return " / ".join(top[:4])

# =========================
# Extractors
# =========================
class BaseExtractor:
    exts: Tuple[str,...] = ()
    def can_handle(self, p:Path)->bool: return p.suffix.lower() in self.exts
    def extract(self, p:Path)->Dict[str,Any]: raise NotImplementedError

class HwpExtractor(BaseExtractor):
    exts=(".hwp",)
    def extract(self, p:Path)->Dict[str,Any]:
        if textract:
            try:
                t = textract.process(str(p)).decode("utf-8","ignore")
                return {"ok":True,"text":TextCleaner.clean(t),"meta":{"engine":"textract"}}
            except Exception: pass
        if platform.system().lower().startswith("win") and win32com:
            try:
                return {"ok":True,"text":"","meta":{"engine":"win32com-hwp"}}
            except Exception: pass
        return {"ok":False,"text":"","meta":{"error":"HWP extract failed"}}

class DocDocxExtractor(BaseExtractor):
    exts=(".doc",".docx")
    def extract(self, p:Path)->Dict[str,Any]:
        if p.suffix.lower()==".docx" and docx:
            try:
                d=docx.Document(str(p))
                t="\n".join(par.text for par in d.paragraphs)
                return {"ok":True,"text":TextCleaner.clean(t),"meta":{"engine":"python-docx","paras":len(d.paragraphs)}}
            except Exception: pass
        if textract:
            try:
                t=textract.process(str(p)).decode("utf-8","ignore")
                return {"ok":True,"text":TextCleaner.clean(t),"meta":{"engine":"textract"}}
            except Exception: pass
        if platform.system().lower().startswith("win") and win32com:
            try:
                return {"ok":True,"text":"","meta":{"engine":"win32com-word"}}
            except Exception: pass
        return {"ok":False,"text":"","meta":{"error":"DOC/DOCX extract failed"}}

class ExcelLikeExtractor(BaseExtractor):
    exts=(".xlsx",".xls",".xlsm",".xlsb",".xltx",".csv")
    def extract(self, p:Path)->Dict[str,Any]:
        if pd is None:
            return {"ok":False,"text":"","meta":{"error":"pandas required"}}
        try:
            if p.suffix.lower()==".csv":
                df=pd.read_csv(p, nrows=200, encoding="utf-8", engine="python")
                txt=self._df_to_text(df)
                return {"ok":True,"text":txt,"meta":{"engine":"pandas","columns":df.columns.tolist(),"rows_preview":min(200,len(df))}}
            eng = "openpyxl" if p.suffix.lower() in (".xlsx",".xlsm",".xltx") else ("xlrd" if p.suffix.lower()==".xls" else "pyxlsb")
            sheets = pd.read_excel(p, sheet_name=None, nrows=200, engine=eng)
            parts=[]
            for s,df_sheet in sheets.items():
                parts.append(f"[Sheet:{s}]")
                parts.append(" | ".join(map(str, df_sheet.columns.tolist())))
                for _,row in df_sheet.head(50).iterrows():
                    parts.append(" • "+" | ".join(map(lambda x: str(x), row.tolist())))
            return {"ok":True,"text":TextCleaner.clean("\n".join(parts)),"meta":{"engine":"pandas","sheets":list(sheets.keys())}}
        except Exception as e:
            return {"ok":False,"text":"","meta":{"error":f"excel/csv read failed: {e}"}}
    @staticmethod
    def _df_to_text(df)->str:
        cols=" | ".join(map(str, df.columns.tolist()))
        rows=[]
        for _,row in df.head(50).iterrows():
            rows.append(" • "+" | "+" | ".join(map(lambda x: str(x), row.tolist())))
        return TextCleaner.clean(f"{cols}\n"+"\n".join(rows))

class PdfExtractor(BaseExtractor):
    exts=(".pdf",)
    def extract(self, p:Path)->Dict[str,Any]:
        if pdfminer_extract_text:
            try:
                t=pdfminer_extract_text(str(p))
                return {"ok":True,"text":TextCleaner.clean(t),"meta":{"engine":"pdfminer"}}
            except Exception: pass
        if textract:
            try:
                t=textract.process(str(p)).decode("utf-8","ignore")
                return {"ok":True,"text":TextCleaner.clean(t),"meta":{"engine":"textract"}}
            except Exception: pass
        return {"ok":False,"text":"","meta":{"error":"PDF extract failed"}}

class PptExtractor(BaseExtractor):
    exts=(".ppt",".pptx")
    def extract(self, p:Path)->Dict[str,Any]:
        if p.suffix.lower()==".pptx" and pptx:
            try:
                pres=pptx.Presentation(str(p))
                texts=[]
                for i,slide in enumerate(pres.slides,1):
                    parts=[]
                    for sh in slide.shapes:
                        if hasattr(sh,"text") and (sh.text or "").strip():
                            parts.append(sh.text)
                    if parts:
                        texts.append(f"[Slide {i}] "+" ".join(parts))
                return {"ok":True,"text":TextCleaner.clean("\n".join(texts)),"meta":{"engine":"python-pptx","slides":len(pres.slides)}}
            except Exception: pass
        if textract:
            try:
                t=textract.process(str(p)).decode("utf-8","ignore")
                return {"ok":True,"text":TextCleaner.clean(t),"meta":{"engine":"textract"}}
            except Exception: pass
        return {"ok":False,"text":"","meta":{"error":"PPT/PPTX extract failed"}}

EXTRACTORS=[HwpExtractor(), DocDocxExtractor(), ExcelLikeExtractor(), PdfExtractor(), PptExtractor()]
EXT_MAP={e:ex for ex in EXTRACTORS for e in ex.exts}

# =========================
# 코퍼스 빌더 (요약/제목 포함)
# =========================
@dataclass
class ExtractRecord:
    path:str; ext:str; ok:bool; text:str; meta:Dict[str,Any]
    size:Optional[int]=None; mtime:Optional[float]=None
    summary:str=""; title:str=""

class CorpusBuilder:
    def __init__(self, max_text_chars:int=200_000, progress:bool=True, translate:bool=False):
        self.max_text_chars=max_text_chars
        self.progress=progress
        self.translate = translate  # 호환 필드(현재 번역 미사용)

    def build(self, file_rows:List[Dict[str,Any]]):
        if pd is None:
            raise RuntimeError("pandas 필요. pip install pandas")
        total=len(file_rows)
        recs:List[ExtractRecord]=[]

        if self.progress and tqdm:
            it = tqdm(file_rows, desc="📥 Extract + Summarize", unit="file")
            for row in it:
                recs.append(self._extract_one(row))
            it.close()
        else:
            print("📥 Extract 시작", flush=True)
            prog=ProgressLine(total, "extracting", update_every=max(1,total//100 or 1))
            for row in file_rows:
                recs.append(self._extract_one(row))
                prog.update(1)
            prog.close()

        df = pd.DataFrame([r.__dict__ for r in recs])
        ok = int(df["ok"].sum()) if len(df)>0 else 0
        fail = int((~df["ok"]).sum()) if len(df)>0 else 0
        print(f"✅ Extract 완료: ok={ok}, fail={fail}", flush=True)
        return df

    def _extract_one(self, row:Dict[str,Any])->ExtractRecord:
        p=Path(row["path"]); ext=p.suffix.lower()
        ex=EXT_MAP.get(ext)
        if not ex:
            return ExtractRecord(str(p), ext, False, "", {"error":"no extractor"},
                                 row.get("size"), row.get("mtime"), "", "")
        try:
            out=ex.extract(p)
            full_text=(out.get("text","") or "")[:self.max_text_chars]
            summ = summarize_extractive(full_text, max_sentences=3) if full_text else ""
            title = make_title_like(full_text, fallback=p.stem)
            return ExtractRecord(str(p), ext, bool(out.get("ok",False)), full_text, out.get("meta",{}),
                                 row.get("size"), row.get("mtime"), summ, title)
        except Exception as e:
            return ExtractRecord(str(p), ext, False, "", {"error":f"extract crash: {e}"},
                                 row.get("size"), row.get("mtime"), "", "")

    @staticmethod
    def save(df, out_path:Path):
        """Parquet 엔진(pyarrow/fastparquet) 없으면 CSV로 자동 폴백."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower()==".parquet":
            try:
                df.to_parquet(out_path, index=False)
                print(f"✅ Parquet 저장: {out_path}")
                return
            except Exception as e:
                csv_path = out_path.with_suffix(".csv")
                df.to_csv(csv_path, index=False, encoding="utf-8")
                print(f"⚠ Parquet 엔진 없음 → CSV로 저장: {csv_path}\n   상세: {e}")
                return
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✅ CSV 저장: {out_path}")

# =========================
# (옵션) 인덱싱 바로 실행
# =========================
def run_indexing(corpus_path: Path, cache_dir: Path):
    """
    corpus_path(.parquet|.csv)을 로드해 의미 임베딩 인덱스를 생성/갱신.
    Retriever.ready(rebuild=True)로 항상 새로 만들고 캐시에 저장.
    """
    print("🚀 Starting semantic indexing...")
    from retriever import Retriever  # 지연 임포트(순환 방지)
    retriever = Retriever(corpus_path=corpus_path, cache_dir=cache_dir)
    retriever.ready(rebuild=True)
    print("✨ Indexing successfully completed.")
