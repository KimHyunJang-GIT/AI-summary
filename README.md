# InfoPilot: AI 기반 문서 검색 엔진

InfoPilot은 로컬 PC에 저장된 다양한 형식의 문서(한글, 워드, 파워포인트, PDF, 엑셀 등)를 스캔하고, 그 내용을 기반으로 의미 기반 검색을 수행하는 데스크톱 애플리케이션입니다. 자연어 질의를 통해 사용자가 원하는 문서를 빠르고 정확하게 찾을 수 있도록 돕습니다.

## 주요 기능

-   **다양한 문서 형식 지원**: hwp, docx, pptx, pdf, xlsx 등 주요 문서 형식의 텍스트 추출.
-   **의미 기반 검색**: 자연어 질의를 통해 문서의 의미적 유사도를 분석하여 관련성 높은 문서를 검색.
-   **학습 데이터 관리**:
    -   **전체 학습**: PC 전체 드라이브를 스캔하여 새로운 코퍼스(문서 집합)와 벡터 인덱스를 구축.
    -   **부분 업데이트**: 기존 데이터를 유지한 채, 새로 추가되거나 수정된 파일만 효율적으로 업데이트.
-   **데스크톱 애플리케이션**: CustomTkinter 기반의 직관적이고 빠른 사용자 인터페이스 제공.

## 프로젝트 구조 설명

InfoPilot 프로젝트는 다음과 같은 주요 디렉토리와 파일로 구성되어 있습니다.

-   `src/`: 애플리케이션의 핵심 로직을 포함합니다.
    -   `src/config.py`: 모델 이름, 경로, 제외 디렉토리, 지원 확장자 등 모든 전역 설정 변수를 정의합니다.
    -   `src/core/`: 문서 처리, 인덱싱, 검색 등 핵심 기능을 담당하는 모듈들이 위치합니다.
        -   `src/core/corpus.py`: 문서에서 텍스트를 추출하고 코퍼스(문서 집합)를 구축하는 로직.
        -   `src/core/indexing.py`: 구축된 코퍼스를 기반으로 벡터 인덱스를 생성하고 관리하는 로직.
        -   `src/core/retrieval.py`: 벡터 인덱스를 사용하여 실제 검색 질의를 처리하는 로직.
        -   `src/core/helpers.py`: 드라이브 목록 가져오기, 질의 파싱 등 공통적으로 사용되는 유틸리티 함수들.
        -   `src/core/extract.py`: 각 파일 형식별 텍스트 추출기 정의.
    -   `src/app/`: 애플리케이션의 주요 기능 모듈을 포함합니다.
        -   `src/app/chat.py`: 채팅 인터페이스와 관련된 핵심 로직을 담당합니다.
    -   `src/cli/`: 커맨드 라인 인터페이스(CLI) 관련 코드를 포함합니다.
        -   `src/cli/main.py`: CLI 애플리케이션의 메인 진입점입니다.
-   `ui/`: 사용자 인터페이스(UI) 관련 코드를 포함합니다.
    -   `ui/app.py`: 애플리케이션의 메인 진입점이며, CustomTkinter 기반의 주 창과 화면 전환 로직을 관리합니다.
    -   `ui/screens/`: 애플리케이션의 각 화면(스크린)을 정의하는 모듈들이 위치합니다.
        -   `ui/screens/home_screen.py`: 앱 시작 시 표시되는 홈 화면.
        -   `ui/screens/chat_screen.py`: 자연어 질의를 입력하고 검색 결과를 확인하는 채팅 화면.
        -   `ui/screens/train_screen.py`: 전체 문서 학습 과정을 시작하고 진행 상황을 모니터링하는 화면.
        -   `ui/screens/update_screen.py`: 기존 데이터를 업데이트하는 과정을 시작하고 모니터링하는 화면.
-   `data/`: 스캔된 파일 목록(`found_files.csv`), 추출된 문서 코퍼스(`corpus.parquet`) 등 애플리케이션 데이터가 저장됩니다.
-   `models/`: SentenceTransformer 모델(`LaBSE`) 파일 및 학습 메타 정보(`topic_model.joblib`)가 저장됩니다.
-   `index_cache/`: 벡터 인덱스 캐시 파일이 저장됩니다.

## UI 설명 (CustomTkinter)

InfoPilot의 UI는 Python의 `CustomTkinter` 라이브러리를 사용하여 개발되었습니다. 이는 웹 기반 UI(예: Streamlit)와 달리 별도의 웹 서버 없이 독립적인 데스크톱 애플리케이션으로 실행되며, 더 빠르고 안정적인 사용자 경험을 제공합니다.

-   **`ui/app.py`**: 애플리케이션의 메인 창(`ctk.CTk`)을 생성하고, 왼쪽 사이드바에 홈, 채팅, 학습, 업데이트 화면으로 이동하는 내비게이션 버튼을 배치합니다. `select_frame()` 메서드를 통해 각 화면 간의 전환을 관리합니다.
-   **`ui/screens/` 디렉토리 내 각 화면 모듈**: `ctk.CTkFrame`을 상속받아 각 화면의 고유한 UI 요소와 로직을 구현합니다. `refresh_state()` 메서드를 통해 화면이 활성화될 때마다 최신 데이터 상태를 반영하도록 설계되었습니다.

## LaBSE 모델 설치 안내

InfoPilot은 의미 기반 검색을 위해 `sentence-transformers/LaBSE` 모델을 사용합니다. 이 모델은 애플리케이션 실행 시 자동으로 다운로드되지만, 특정 환경에서는 수동 설치가 필요할 수 있습니다.

### 1. 자동 다운로드 (권장)

-   InfoPilot 애플리케이션을 처음 실행하거나, `models/` 디렉토리에 `LaBSE` 모델이 없는 경우, 애플리케이션이 자동으로 Hugging Face Hub에서 모델을 다운로드합니다.
-   **주의**: 모델 크기(약 500MB)로 인해 초기 다운로드에 시간이 걸릴 수 있으며, 안정적인 인터넷 연결이 필요합니다.

### 2. 수동 다운로드 및 배치 (선택 사항, 문제 발생 시)

네트워크 문제 등으로 자동 다운로드가 어려운 경우, 다음 절차에 따라 모델을 수동으로 다운로드하여 배치할 수 있습니다.

1.  **모델 다운로드**:
    -   웹 브라우저를 열고 다음 Hugging Face 모델 페이지로 이동합니다: [https://huggingface.co/sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE)
    -   페이지 상단의 **"Files and versions"** 탭을 클릭합니다.
    -   모델 파일들(예: `pytorch_model.bin`, `config.json`, `tokenizer.json` 등)을 모두 다운로드합니다. 일반적으로 "Download all files" 옵션이 제공됩니다.

2.  **디렉토리 구조 생성**:
    -   프로젝트 루트 디렉토리(`InfoPilot/`) 아래에 `models/models--sentence-transformers--LaBSE/snapshots/` 경로를 생성합니다.
    -   다운로드 페이지에서 모델 버전(커밋 해시)을 확인하여 `<commit_hash>` 부분을 대체합니다. (예: `models/models--sentence-transformers--LaBSE/snapshots/5f4e1a.../`)
    -   **예시 경로**: `InfoPilot/models/models--sentence-transformers--LaBSE/snapshots/5f4e1a.../`

3.  **파일 배치**:
    -   다운로드한 모든 모델 파일들을 위에서 생성한 `<commit_hash>` 디렉토리 안에 배치합니다.

## 실행 방법

1.  **의존성 설치:**
    프로젝트 루트 디렉토리에서 터미널을 열고 다음 명령어를 실행하여 필요한 라이브러리들을 설치합니다.

    ```bash
    pip install -r requirements.txt
    pip install hf_xet
    ```

2.  **애플리케이션 실행:**
    설치 완료 후, 다음 명령어를 실행하여 InfoPilot 애플리케이션을 시작합니다.

    ```bash
    python ui/app.py
    ```

    애플리케이션이 실행되면 데스크톱 창이 나타나며, 모든 기능에 접근할 수 있습니다.
