# InfoPilot UI 클라이언트

이 디렉토리에는 InfoPilot 애플리케이션의 사용자 인터페이스(UI) 관련 코드가 포함되어 있습니다. Python의 `CustomTkinter` 라이브러리를 기반으로 개발되었으며, 모든 핵심 기능은 FastAPI 백엔드 API와 HTTP 통신을 통해 상호작용합니다.

## 주요 특징

-   **CustomTkinter 기반**: 현대적이고 커스터마이징 가능한 데스크톱 UI를 제공합니다.
-   **클라이언트-서버 아키텍처**: `requests` 라이브러리를 사용하여 FastAPI 백엔드(`src/api`)와 통신하며, UI와 비즈니스 로직이 명확하게 분리되어 있습니다.
-   **비동기 작업 처리**: 장시간이 소요되는 백엔드 API 호출은 별도의 스레드에서 처리하여 UI의 응답성을 유지합니다.
-   **상태 관리**: `src/core/helpers.py`의 `have_all_artifacts()` 함수를 통해 학습 데이터 유무를 확인하고, UI 상태를 동적으로 변경하여 사용자에게 적절한 안내를 제공합니다.

## 디렉토리 구조

-   `ui/app.py`: 애플리케이션의 메인 진입점입니다. CustomTkinter의 `CTk` 클래스를 상속받아 주 창을 생성하고, 왼쪽 사이드바에 내비게이션 버튼을 배치합니다. `select_frame()` 메서드를 통해 각 화면(`screens`) 간의 전환을 관리하며, `start_task()` 및 `end_task()` 콜백을 통해 백엔드 작업 중 UI 버튼의 활성화/비활성화를 제어합니다.

-   `ui/screens/`: 애플리케이션의 각 화면(스크린)을 정의하는 모듈들이 위치합니다. 각 화면은 `ctk.CTkFrame`을 상속받아 고유한 UI 요소와 로직을 구현합니다.
    -   `ui/screens/home_screen.py`: 앱 시작 시 표시되는 기본 홈 화면입니다.
    -   `ui/screens/chat_screen.py`: 자연어 질의를 입력하고 검색 결과를 확인하는 채팅 인터페이스를 제공합니다. **BM25 및 재랭킹 옵션을 UI에서 직접 설정하여 검색에 반영할 수 있습니다.** `/chat` FastAPI 엔드포인트와 통신합니다.
    -   `ui/screens/train_screen.py`: 전체 문서 학습 과정을 시작하고 진행 상황을 모니터링하는 화면입니다. `/scan` 및 `/train` FastAPI 엔드포인트와 통신합니다.
    -   `ui/screens/update_screen.py`: 기존 데이터를 업데이트하는 과정을 시작하고 모니터링하는 화면입니다. `/update` FastAPI 엔드포인트와 통신합니다.

## 백엔드 통신

모든 UI 화면은 `src/config.py`에 정의된 `FASTAPI_URL`을 기반으로 `requests` 라이브러리를 사용하여 FastAPI 백엔드와 통신합니다. API 호출은 `threading.Thread`를 통해 비동기적으로 실행되어 UI가 블로킹되지 않도록 합니다.

예시 (`ui/screens/chat_screen.py`에서 `/chat` 엔드포인트 호출):

```python
import requests
from src.config import FASTAPI_URL

# ... (중략) ...

            payload = {
                "query": cleaned_query,
                "topk": DEFAULT_TOP_K,
                "filters": filters,
                "use_bm25": self.bm25_checkbox.get() == 1,
                "use_reranker": self.reranker_checkbox.get() == 1
            }
            
            chat_url = f"{FASTAPI_URL}/chat"
            response = requests.post(chat_url, json=payload)
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생
            data = response.json()

# ... (중략) ...
```

이 `README.md`는 InfoPilot UI 클라이언트의 구조와 작동 방식을 이해하는 데 도움이 됩니다.
