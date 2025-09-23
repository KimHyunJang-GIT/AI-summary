# -*- coding: utf-8 -*-
"""프로젝트 전반에서 사용되는 공통 헬퍼 함수 모음."""
import os
import string
import re
from pathlib import Path # Path 임포트 추가

from src.config import BASE_EXT_MAP, CORPUS_PARQUET, TOPIC_MODEL_PATH # config에서 필요한 경로 임포트

# 위치 키워드와 실제 경로 문자열 매핑
LOCATION_MAP = {
    "다운로드": "Downloads",
    "바탕화면": "Desktop",
}

def get_drives():
    """시스템에 존재하는 드라이브 목록을 반환합니다."""
    drives = []
    for letter in string.ascii_uppercase:
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            drives.append(drive)
    return drives

def parse_query_and_filters(query: str) -> tuple[str, dict]:
    """사용자 질의에서 검색어, 명시적/암시적 필터를 분리합니다."""
    filters = {}
    temp_query = f" {query} " # 단어 경계(" ")를 위해 앞뒤에 공백 추가

    # 1. 명시적 필터 추출 (예: ext:pdf, title:보고서)
    explicit_filter_pattern = re.compile(r'(\w+):([^\s]+)')
    for match in reversed(list(explicit_filter_pattern.finditer(temp_query))):
        filters[match.group(1).lower()] = match.group(2)
        temp_query = temp_query[:match.start()] + temp_query[match.end():]

    # 2. 암시적 위치 필터 추출 (예: 다운로드에 있는, 바탕화면의)
    # 위치 키워드 뒤에 오는 조사를 유연하게 처리 (에 있는, 의, 폴더의 등)
    for keyword, path_part in LOCATION_MAP.items():
        location_pattern = re.compile(f' {re.escape(keyword)}(?:에|의|폴더(?:의|에)?)? ')
        if match := location_pattern.search(temp_query):
            filters['path'] = path_part
            temp_query = temp_query.replace(match.group(0), " ", 1)
            break # 첫 번째 위치 키워드만 사용

    # 3. 암시적 확장자 필터 추출 (예: pdf, 엑셀 파일)
    ext_map = {keyword: ext for ext, keywords in BASE_EXT_MAP.items() for keyword in keywords}
    ext_map.update({ext: ext for ext in BASE_EXT_MAP})
    
    # 직접적인 확장자 (.pdf) 먼저 처리
    direct_ext_pattern = re.compile(r'\.(\w+)\b', re.IGNORECASE)
    if match := direct_ext_pattern.search(temp_query):
        if (matched_ext := "." + match.group(1).lower()) in ext_map:
            filters['ext'] = ext_map[matched_ext]
            temp_query = temp_query.replace(match.group(0), "", 1)
    
    # 키워드 (pdf, 엑셀) 처리
    if 'ext' not in filters:
        sorted_ext_keywords = sorted([k for k in ext_map if not k.startswith('.')], key=len, reverse=True)
        ext_keyword_regex_parts = [re.escape(k) + r'(?:\s*(?:파일|문서|자료))?' for k in sorted_ext_keywords]
        if ext_keyword_regex_parts:
            implicit_ext_keyword_pattern = re.compile(r'\b(' + '|'.join(ext_keyword_regex_parts) + r')\b', re.IGNORECASE)
            if match := implicit_ext_keyword_pattern.search(temp_query):
                matched_keyword = re.sub(r'\s*(?:파일|문서|자료)$', '', match.group(1).lower())
                if matched_keyword in ext_map:
                    filters['ext'] = ext_map[matched_keyword]
                    temp_query = temp_query.replace(match.group(0), "", 1)

    # 4. 암시적 제목 필터 추출 (예: 제목이 보고서인)
    implicit_title_patterns = [re.compile(r'(제목이|이름이)\s*(\S+)(?:인|인문서|인파일)?'), re.compile(r'(\S+)(?:라는|이라는)\s*(제목의|이름의)')]
    for pattern in implicit_title_patterns:
        if match := pattern.search(temp_query):
            filters['title'] = match.group(2) if len(match.groups()) >= 2 else match.group(1)
            temp_query = temp_query.replace(match.group(0), "", 1)
            break

    cleaned_query = re.sub(r'\s+', ' ', temp_query).strip()
    return cleaned_query, filters

def have_all_artifacts() -> bool:
    """필수 파일(코퍼스, 모델)이 모두 존재하는지 확인합니다."""
    return CORPUS_PARQUET.exists() and TOPIC_MODEL_PATH.exists()
