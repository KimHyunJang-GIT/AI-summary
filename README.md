# AI-summary
AI 요약 소프트웨어를 구축하고 학습·조회 파이프라인을 돌리기 위한 저장소입니다.

## 프로젝트 개요
- `infopilot.py` : `scan → train → chat` 절차를 오케스트레이션하는 CLI 진입점
- `filefinder.py` : 파일 시스템을 스캔해 후보 목록을 CSV로 저장
- `pipeline.py` : 코퍼스 정제 및 토픽 모델 학습, Parquet/CSV 아티팩트 생성
- `retriever.py / lnp_chat.py` : 학습된 모델을 불러와 쿼리 검색 및 대화 인터페이스 제공
- `data/`, `index_cache/` : 생성된 코퍼스와 캐시가 위치하는 디렉터리 (대용량 파일은 커밋 금지)

## 핵심 명령
| 단계 | 명령 | 설명 |
| --- | --- | --- |
| 스캔 | `python infopilot.py scan --out data/found_files.csv` | 새/변경 파일 메타데이터 수집 |
| 학습 | `python infopilot.py train --scan_csv data/found_files.csv --corpus data/corpus.parquet` | 코퍼스 및 토픽 모델 생성 |
| 대화 | `python infopilot.py chat --model data/topic_model.joblib --corpus data/corpus.parquet --cache index_cache` | 의미 기반 검색 챗봇 실행 (`--translate`, `--show-translation`, `--lexical-weight` 옵션 지원) |
| 감시 | `python infopilot.py watch --cache index_cache --corpus data/corpus.parquet --model data/topic_model.joblib` | 파일 변경을 감지해 코퍼스·FAISS 인덱스를 실시간으로 증분 갱신 |

파이프라인실행
python infopilot.py pipeline --corpus data/corpus.parquet --model data/topic_model.joblib




## OS별 설치 및 실행 가이드
### macOS
1. Homebrew 등으로 Python 3.9 이상이 설치되어 있는지 확인 (`python3 --version`).
2. 가상환경 생성 및 활성화:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. 의존성 설치:
   ```bash
   python3 -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Parquet 의존성(예: `fastparquet`) 빌드 에러가 나면 Xcode Command Line Tools를 설치하세요.
   ```bash
   xcode-select --install
   ```
5. 필요 시 명령을 실행합니다 (예: `pytest -q`, `python infopilot.py train ...`).

### Linux / WSL
1. 필수 빌드 도구 준비(Parquet 가속을 위해 권장):
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential libsnappy-dev
   ```
2. 가상환경 생성 및 활성화:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. 의존성 설치:
   ```bash
   python3 -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *시스템 Python을 그대로 사용할 경우 Debian/Ubuntu는 `--break-system-packages` 옵션이 필요할 수 있습니다.*
4. 테스트/학습/대화를 필요한 순서대로 실행합니다.

### Windows (PowerShell)
1. 64비트 Python 3.9 이상 설치 여부 확인 (`python --version`).
단계별 절차 (윈도우 PowerShell 기준)

기존 venv 삭제

Remove-Item .venv -Recurse -Force


 2. 가상환경 생성 및 활성화: 
   ```powershell
    python -m venv .venv
   .\.venv\Scripts\Activate.ps1 
   ```
3. 의존성 설치:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. `fastparquet` 설치가 실패하면 Microsoft C++ Build Tools를 설치하거나 최신 휠이 포함된 Python 버전을 사용하세요.
5. 이후 `pytest -q`, `python infopilot.py scan ...` 등 필요한 명령을 순차 실행합니다.

#### Windows 실행 파일(.exe) 빌드
PowerShell에서 아래 순서로 실행하면 `dist/InfoPilotLauncher.exe`가 생성됩니다.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt pyinstaller
powershell -ExecutionPolicy Bypass -File scripts\build_windows_exe.ps1
```

빌드된 파일을 더블 클릭하면 간단한 콘솔 메뉴가 열리고, 파이프라인 실행(옵션: 번역, 루트 경로 입력) 후 Chat 자동 실행 또는 Chat만 단독 실행을 선택할 수 있습니다.

### Windows (Command Prompt)
1. 가상환경 생성 및 활성화:
   ```cmd
   python -m venv .venv
   .\.venv\Scripts\activate.bat
   ```
2. 의존성 설치:
   ```cmd
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. 명령 실행 예시:
   ```cmd
   pytest -q
   python infopilot.py chat --model data\topic_model.joblib --corpus data\corpus.parquet --cache index_cache --translate --show-translation
   ```

## 테스트 및 검증

### Benchmark & Evaluation
1. ANN vs 정확 검색 벤치마크
   ```bash
   source .venv/bin/activate
   python benchmarks/ann_benchmark.py \
     --doc-count 20000 --queries 100 --top-k 10 --ann-threshold 2000 \
     --ef-search 128 --ef-construction 200 --ann-m 32 \
     --target-overlap 0.95 --target-p95 500 --output benchmarks/results/ann_run.json
   ```
   - `overlap@k`(정확 검색과의 겹침)과 `ann_ms.p95`가 목표에 못 미치면 종료 코드가 1이므로 CI 회귀 검증에 사용할 수 있습니다.
2. 정확도 지표(P@K, nDCG@K) 측정
   ```bash
   python benchmarks/accuracy_eval.py \
     --labels benchmarks/fixtures/sample_labels.csv \
     --predictions benchmarks/fixtures/sample_predictions.csv \
     --k 1 5 --target-p 0.8 --target-ndcg 0.7 --output benchmarks/results/accuracy.json
   ```
   - 실측 라벨/예측 CSV로 바꿔 실행하고, 목표치를 벗어나면 비정상 종료로 회귀를 잡아낼 수 있습니다.
3. 프런트엔드 스모크
   ```bash
   pytest -m smoke
   ```
   - 세션/ANN 경로 관련 테스트가 포함되어 프런트 버튼-API 훅이 정상 동작하는지 재확인합니다.

### 새 UI 이벤트 로그 확인
로컬에서 카드 버튼을 클릭하면 `localStorage`의 `infopilot_ui_events` 키에 성공/실패 이벤트가 쌓입니다. 콘솔에서
```js
JSON.parse(localStorage.getItem('infopilot_ui_events') || '[]')
```
을 실행해 API 전송 누락 여부를 빠르게 점검하세요.

- 모든 플랫폼에서 `pytest -q`로 스모크 + 풀 테스트를 한 번에 수행하세요.
- 빠른 확인만 필요하면 `pytest -q -m smoke`, 전체 회귀만 재실행하려면 `pytest -q -m full`을 사용할 수 있습니다.
- `tests/conftest.py`가 런타임에 필요한 `JOBLIB_MULTIPROCESSING=0`과 임시 폴더 설정을 제공하므로 추가 환경 변수 설정이 필요 없습니다.
- 데이터 파이프라인 변경 시 `scan → train`을 재실행한 뒤 캐시 무결성을 확인하고, `chat`에서 결과를 수동 검증하세요.

## 문제 해결 팁
- Parquet 저장 실패 시 `requirements.txt`에 포함된 `fastparquet`가 제대로 설치되었는지 확인하고, 빌드 도구나 네트워크 제한을 점검합니다. 필요한 경우 `pyarrow`와 같은 대체 엔진을 수동으로 설치해도 됩니다.
- 검색은 기본적으로 다국어 Sentence-BERT 의미 검색으로 동작합니다. 필요 시 `--lexical-weight`를 0 초과 값으로 지정해 BM25 보조 가중치를 추가하세요.
- 번역이 필요하면 `--translate`(질문 번역) 또는 `--show-translation`(결과 미리보기 번역) 옵션을 추가하고, 오류 발생 시 `deep-translator` 패키지와 외부 네트워크 접근성을 점검하세요.
- 실시간 반영이 필요하면 `python infopilot.py watch ...`로 watcher를 띄워 새/수정된 파일을 자동 추출·임베딩하세요.

깃허브 푸쉬방법

1. Git 상태 확인
git status


→ 수정된 파일/추가된 파일 목록 확인

2. 변경 파일 스테이징
git add -A


(전체 변경 사항을 스테이징)

3. 커밋 생성
git commit -m "파이프라인 실행 코드 및 테스트 통과 후 정리"


→ 메시지는 원하는 걸로 자유롭게 작성 가능

4. 원격 브랜치 확인
git branch -vv


→ 현재 작업 브랜치(main, test1 등)와 연결된 원격(remote/origin) 확인

5. 푸시
git push origin main


만약 현재 브랜치가 main이 아니라면, 예를 들어 test1이라면:

git push origin test1

6. (옵션) 강제 푸시

히스토리가 꼬였거나, 기존 커밋을 덮어써야 하는 경우만:

git push origin main --force


⚠️ 하지만 협업 환경에서는 권장하지 않습니다. 본인만 쓰는 저장소일 때만 사용하세요.
>>>>>>> recover-mywork
