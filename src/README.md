
# infopilot_split

원본 파일을 실제로 분해하여 역할별 모듈로 구성했습니다.
- 일부 심볼은 원본으로부터 **코드 블록을 추출**하여 파일에 직접 포함했고,
- 찾을 수 없는 심볼은 **원본에서 임시 재수출**(fallback)했습니다.

## 구조
- core/file_finder.py — FileFinder, StartupSpinner
- core/extract.py — TextCleaner, BaseExtractor, HwpExtractor, DocxExtractor, PdfExtractor, PptxExtractor, simple_summary
- core/corpus.py — ExtractRecord, CorpusBuilder
- core/indexing.py — run_indexing
- core/index_store.py — IndexPaths, VectorIndex
- core/retrieval.py — Retriever
- app/chat.py — ChatTurn, LNPChat
- app/paths.py — 공통 경로/아티팩트 체크
- cli/infopilot_cli.py — cmd_scan, cmd_train, cmd_chat, main
