# pipeline.py  (Step2: 추출 + 학습)
import os, re, sys, time, threading, platform
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

# ---- 선택 의존성(있으면 사용) ----
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None
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
    import joblib
except Exception:
    joblib = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.pipeline import Pipeline
except Exception:
    TfidfVectorizer = TruncatedSVD = MiniBatchKMeans = Pipeline = None

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

TOKEN_PATTERN = r'(?u)(?:[가-힣]{1,}|[A-Za-z0-9]{2,})'


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
                return {"ok":True,"text":txt,"meta":{"engine":"pandas","columns":df.columns.tolist(), "rows_preview":min(200,len(df))}}
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
            rows.append(" • "+" | ".join(map(lambda x: str(x), row.tolist())))
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
# 코퍼스 빌더 (번역 기능 추가)
# =========================
@dataclass
class ExtractRecord:
    path:str; ext:str; ok:bool; text:str; meta:Dict[str,Any]; size:Optional[int]=None; mtime:Optional[float]=None

class CorpusBuilder:
    def __init__(self, max_text_chars:int=200_000, progress:bool=True, translate:bool=False):
        self.max_text_chars=max_text_chars
        self.progress=progress
        self.translate = translate
        self.translator = GoogleTranslator(source='auto', target='en') if translate and GoogleTranslator else None
        if translate and not self.translator:
            print("⚠️ 경고: 'deep-translator' 라이브러리를 찾을 수 없어 번역 기능이 비활성화됩니다.")
            print("   해결: pip install -U deep-translator")

    def build(self, file_rows:List[Dict[str,Any]]):
        if pd is None: raise RuntimeError("pandas 필요. pip install pandas")
        total=len(file_rows)
        recs:List[ExtractRecord]=[]

        iterator = file_rows
        if self.progress and tqdm:
            desc = "📥 Extract & Translate" if self.translate else "📥 Extract"
            iterator = tqdm(file_rows, desc=desc, unit="file")
        else:
            print("📥 Extract 시작", flush=True)
            prog=ProgressLine(total, "extracting", update_every=max(1,total//100 or 1))

        for row in iterator:
            recs.append(self._extract_one(row))
            if not (self.progress and tqdm):
                prog.update(1)
        
        if self.progress and tqdm:
            iterator.close()
        else:
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
            return ExtractRecord(str(p), ext, False, "", {"error":"no extractor"}, row.get("size"), row.get("mtime"))
        try:
            out=ex.extract(p)
            original_text=(out.get("text","") or "")[:self.max_text_chars]

            # [번역 기능]
            text_for_model = original_text
            if self.translator and original_text.strip():
                try:
                    text_for_model = self.translator.translate(original_text)
                except Exception as e:
                    print(f"\n[경고] '{p.name}' 번역 실패. 원본 텍스트 사용. 오류: {e}")

            return ExtractRecord(str(p), ext, bool(out.get("ok",False)), text_for_model, out.get("meta",{}), row.get("size"), row.get("mtime"))
        except Exception as e:
            return ExtractRecord(str(p), ext, False, "", {"error":f"extract crash: {e}"}, row.get("size"), row.get("mtime"))

    @staticmethod
    def save(df, out_path:Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix.lower()
        if ext == ".parquet":
            try:
                df.to_parquet(out_path, index=False)
                print(f"✅ Parquet 저장: {out_path}")
                return
            except Exception as e:
                csv_path = out_path.with_suffix(".csv")
                df.to_csv(csv_path, index=False, encoding="utf-8")
                print(f"⚠️ Parquet 엔진 없음 → CSV로 저장: {csv_path}\n   상세: {e}")
                return
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✅ CSV 저장: {out_path}")


# =========================
# 토픽 모델
# =========================
@dataclass
class TrainConfig:
    max_features:int=200_000
    n_components:int=256
    n_clusters:int=30
    ngram_range:Tuple[int,int]=(1,2)
    min_df:int=2
    max_df:float=0.8

class TopicModel:
    def __init__(self, cfg:TrainConfig):
        if any(x is None for x in (TfidfVectorizer, TruncatedSVD, MiniBatchKMeans, Pipeline)):
            raise RuntimeError("scikit-learn 필요. pip install scikit-learn joblib")
        self.cfg=cfg
        self.pipeline:Optional[Pipeline]=None

    def fit(self, df, text_col="text"):
        texts=(df[text_col].fillna("").astype(str)).tolist()
        print("🧠 학습 준비: TF-IDF → SVD → KMeans", flush=True)
        spin=Spinner(prefix="  학습 중")
        spin.start()
        try:
            self.pipeline = Pipeline(steps=[
                ("tfidf", TfidfVectorizer(
                    token_pattern=TOKEN_PATTERN,
                    ngram_range=self.cfg.ngram_range,
                    max_features=self.cfg.max_features,
                    min_df=self.cfg.min_df,
                    max_df=self.cfg.max_df,
                )),
                ("svd", TruncatedSVD(n_components=self.cfg.n_components, random_state=42)),
                ("kmeans", MiniBatchKMeans(n_clusters=self.cfg.n_clusters, random_state=42, batch_size=2048, n_init="auto")),
            ])
            t0=time.time()
            self.pipeline.fit(texts)
            t1=time.time()
        finally:
            spin.stop()
        print(f"✅ 학습 완료 (docs={len(texts):,}, {t1-t0:.1f}s)", flush=True)
        return self

    def predict(self, df, text_col="text")->List[int]:
        texts=(df[text_col].fillna("").astype(str)).tolist()
        return self.pipeline.predict(texts)

    def transform(self, df, text_col="text"):
        texts=(df[text_col].fillna("").astype(str)).tolist()
        X=self.pipeline.named_steps["tfidf"].transform(texts)
        Z=self.pipeline.named_steps["svd"].transform(X)
        return Z

    def save(self, path:Path):
        if joblib is None: raise RuntimeError("joblib 필요")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"cfg":self.cfg,"pipeline":self.pipeline}, path)


# =========================
# 파이프라인 실행 (메인 함수)
# =========================
def run_step2(file_rows:List[Dict[str,Any]],
              out_corpus:Path=Path("./corpus.parquet"),
              out_model:Path=Path("./topic_model.joblib"),
              cfg:TrainConfig=TrainConfig(),
              use_tqdm:bool=True,
              translate:bool=False): # 번역 옵션 추가
    global tqdm
    if not use_tqdm:
        tqdm=None

    print("=== Step 2 시작: 내용 추출 & 학습 === (번역: " + ("활성" if translate else "비활성") + ")", flush=True)
    t_all=time.time()

    # 1) Extract & Translate
    cb=CorpusBuilder(max_text_chars=200_000, progress=True, translate=translate)
    df=cb.build(file_rows)

    # 2) 학습 데이터 필터
    if pd is None: raise RuntimeError("pandas 필요")
    train_df = df[df["ok"] & (df["text"].str.len()>0)].copy()
    print(f"🧹 학습 대상 문서: {len(train_df):,}/{len(df):,}", flush=True)
    if len(train_df)==0:
        cb.save(df, out_corpus)
        print(f"⚠️ 유효 텍스트 없음. 코퍼스만 저장: {out_corpus}", flush=True)
        return df, None

    # 3) Train
    tm=TopicModel(cfg)
    tm.fit(train_df, text_col="text")
    labels=tm.predict(train_df)
    train_df["topic"]=labels
    df = df.merge(train_df[["path","topic"]], on="path", how="left")

    # 4) 저장
    cb.save(df, out_corpus)
    if joblib:
        tm.save(out_model)

    dt_all=time.time()-t_all
    print(f"💾 저장 완료: corpus → {out_corpus} | model → {out_model}", flush=True)
    print(f"🎉 Step 2 종료 (총 {dt_all:.1f}s)", flush=True)
    return df, tm
