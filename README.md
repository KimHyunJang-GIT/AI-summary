
# InfoPilot Split Bundle

- 역할 분리된 패키지(`infopilot_split/`)와 Streamlit UI(`ui/app.py`)가 포함된 테스트 번들입니다.
- **데이터 존재 여부에 따라** 버튼이 달라집니다:
  - `data/corpus.csv` 또는 `data/corpus.parquet` + `models/topic_model.joblib` 있으면 → **다시 교육하기/채팅창으로 가기**
  - 없으면 → **교육시키기**

## 설치
```bash
pip install -r requirements.txt
```

## 실행
```bash
streamlit run ui/app.py
```

## 경로
- 기본 경로는 현재 작업 디렉터리 기준입니다.
  - `data/` (corpus.*)
  - `models/` (topic_model.joblib)
  - `index_cache/` (벡터 인덱스 캐시)
필요 시 `infopilot_split/app/paths.py`의 `Paths`를 수정하세요.
