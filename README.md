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
| 대화 | `python infopilot.py chat --model data/topic_model.joblib --corpus data/corpus.parquet --cache index_cache` | 검색 기반 챗봇 실행 (`--no-translate` 옵션 지원) |

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
   python infopilot.py chat --model data\topic_model.joblib --corpus data\corpus.parquet --cache index_cache --no-translate
   ```

## 테스트 및 검증
- 모든 플랫폼에서 `pytest -q`로 회귀 테스트를 수행하세요.
- `tests/conftest.py`가 런타임에 필요한 `JOBLIB_MULTIPROCESSING=0`과 임시 폴더 설정을 제공하므로 추가 환경 변수 설정이 필요 없습니다.
- 데이터 파이프라인 변경 시 `scan → train`을 재실행한 뒤 캐시 무결성을 확인하고, `chat`에서 결과를 수동 검증하세요.

## 문제 해결 팁
- Parquet 저장 실패 시 `requirements.txt`에 포함된 `fastparquet`가 제대로 설치되었는지 확인하고, 빌드 도구나 네트워크 제한을 점검합니다. 필요한 경우 `pyarrow`와 같은 대체 엔진을 수동으로 설치해도 됩니다.
- 번역 기능 오류가 발생하면 `deep-translator` 패키지 설치 여부와 외부 네트워크 접근성을 확인하거나 CLI에서 `--no-translate` 옵션으로 비활성화하세요.
