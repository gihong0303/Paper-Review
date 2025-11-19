# FineWeb: decanting the web for the finest text data at scale

**논문 발표**: 2024년
**저자**: Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, Anton Lozhkov, et al.
**소속**: Hugging Face
**논문 링크**: [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
**데이터셋**: [HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

---

## 한 줄 요약
> 96개 Common Crawl 스냅샷에서 15T 토큰의 고품질 영어 웹 데이터를 추출하여, 기존 공개 데이터셋 대비 일관된 성능 향상을 달성한 대규모 데이터 큐레이션

---

## 1. 문제: 웹 데이터의 품질

### 1.1 Common Crawl의 문제점

```
Common Crawl 원본:
- 페타바이트 규모의 웹 크롤
- 품질 매우 들쭉날쭉
- 스팸, 중복, 저품질 콘텐츠 다수

그대로 사용 시:
→ 모델 성능 저하
→ 학습 효율 낮음
```

### 1.2 기존 데이터셋의 한계

| 데이터셋 | 크기 | 문제점 |
|----------|------|--------|
| C4 | 175B | 과도한 필터링 |
| The Pile | 300B | 크기 제한 |
| RefinedWeb | 600B | 비공개 |

---

## 2. FineWeb 파이프라인

### 2.1 전체 구조

```
Common Crawl (96 snapshots)
         ↓
    URL 필터링
         ↓
    텍스트 추출
         ↓
    언어 필터링
         ↓
    품질 필터링
         ↓
    중복 제거
         ↓
   FineWeb (15T tokens)
```

### 2.2 각 단계 상세

#### 1) URL 필터링

```python
# 차단할 URL 패턴
blocked_patterns = [
    "*.porn*",
    "*.xxx*",
    "*spam*",
    "*casino*",
    ...
]

def url_filter(url):
    for pattern in blocked_patterns:
        if matches(url, pattern):
            return False
    return True
```

#### 2) 텍스트 추출 (trafilatura)

```python
from trafilatura import extract

def extract_text(html):
    """trafilatura로 본문 추출"""
    text = extract(
        html,
        include_comments=False,
        include_tables=True,
        deduplicate=True
    )
    return text
```

#### 3) 언어 필터링

```python
from fasttext import load_model

lang_model = load_model("lid.176.bin")

def filter_english(text):
    pred = lang_model.predict(text)
    lang, score = pred[0][0], pred[1][0]

    # 영어이고 확신도 높은 경우만
    if lang == "__label__en" and score > 0.65:
        return True
    return False
```

#### 4) 품질 필터링

```python
def quality_filter(text):
    # 1. 길이 체크
    if len(text) < 200:
        return False

    # 2. 단어 반복 체크
    words = text.split()
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.2:
        return False

    # 3. 특수문자 비율
    special_char_ratio = count_special(text) / len(text)
    if special_char_ratio > 0.3:
        return False

    # 4. 문장 완결성
    if not ends_with_punctuation(text):
        return False

    return True
```

#### 5) 중복 제거 (MinHash)

```python
from datasketch import MinHash, MinHashLSH

def deduplicate_minhash(documents):
    """MinHash LSH로 유사 문서 제거"""
    lsh = MinHashLSH(threshold=0.8, num_perm=128)

    unique_docs = []
    for i, doc in enumerate(documents):
        # MinHash 생성
        mh = MinHash(num_perm=128)
        for word in doc.split():
            mh.update(word.encode('utf8'))

        # 중복 체크
        result = lsh.query(mh)
        if not result:
            lsh.insert(i, mh)
            unique_docs.append(doc)

    return unique_docs
```

---

## 3. FineWeb-Edu: 교육적 콘텐츠

### 3.1 개념

```
FineWeb (15T) → 교육성 점수화 → FineWeb-Edu (1.3T)

가장 교육적인 콘텐츠만 선별
```

### 3.2 교육성 분류기

```python
# LLaMA-3-70B-Instruct로 annotation
prompt = """
Rate the following web page for its educational value
on a scale of 0-5:

0: Not educational (ads, spam, adult content)
1: Minimal educational value
2: Some educational elements
3: Moderately educational
4: Highly educational
5: Outstanding educational content

Text: {text}

Score:
"""

# 점수 4-5인 것만 선별 → FineWeb-Edu
```

### 3.3 효과

| 데이터셋 | 크기 | MMLU 성능 |
|----------|------|-----------|
| FineWeb | 15T | 37.2% |
| FineWeb-Edu | 1.3T | **40.1%** |

**더 적은 데이터로 더 높은 성능!**

---

## 4. 실험 결과

### 4.1 ablation: 각 필터의 기여도

| 설정 | HellaSwag |
|------|-----------|
| 원본 CC | 42.1 |
| + URL 필터 | 44.3 |
| + 품질 필터 | 53.6 |
| + 중복 제거 | 62.8 |
| **FineWeb** | **65.1** |

### 4.2 다른 데이터셋과 비교

1.8B 파라미터 모델, 350B 토큰 학습:

| 데이터셋 | MMLU | HellaSwag | ARC |
|----------|------|-----------|-----|
| C4 | 26.8 | 56.2 | 35.1 |
| The Pile | 28.1 | 57.8 | 36.4 |
| RefinedWeb | 28.7 | 61.3 | 38.2 |
| **FineWeb** | **32.1** | **65.1** | **41.8** |

### 4.3 FineWeb-Edu 효과

| 데이터셋 | MMLU | ARC-C |
|----------|------|-------|
| FineWeb (350B) | 32.1 | 41.8 |
| FineWeb-Edu (350B) | **37.8** | **48.2** |

---

## 5. 핵심 인사이트

### 5.1 필터링의 중요성

```
각 필터링 단계가 모두 중요:

1. URL 필터: 악성 콘텐츠 제거
2. 품질 필터: 노이즈 제거
3. 중복 제거: 학습 효율 향상
4. 교육성 필터: 품질 최적화
```

### 5.2 데이터 품질 vs 양

```
FineWeb: 15T 토큰 → 좋은 성능
FineWeb-Edu: 1.3T 토큰 → 더 좋은 성능

→ 품질이 양보다 중요!
```

### 5.3 중복 제거의 영향

```
중복 제거 전후:
- 성능 +10% 이상 향상
- 학습 수렴 속도 빨라짐
- 암기 대신 일반화
```

---

## 6. 실무 적용 가이드

### 6.1 자체 데이터셋 구축

```python
class DataPipeline:
    def __init__(self):
        self.filters = [
            URLFilter(),
            LanguageFilter(lang="ko", threshold=0.7),
            QualityFilter(min_length=200),
            DeduplicationFilter(threshold=0.8),
            EducationalFilter(min_score=3)  # 선택
        ]

    def process(self, documents):
        for filter in self.filters:
            documents = filter.apply(documents)
            print(f"{filter.name}: {len(documents)} remaining")

        return documents
```

### 6.2 한국어 데이터셋 구축 시

```python
# 한국어 특화 필터
def korean_quality_filter(text):
    # 1. 한국어 비율 체크
    korean_ratio = count_korean(text) / len(text)
    if korean_ratio < 0.5:
        return False

    # 2. 존댓말/반말 일관성
    # 3. 맞춤법 오류 비율
    # 4. 기계 번역 감지

    return True
```

---

## 7. 쉬운 예시

### 7.1 와인 양조 비유

```
FineWeb의 이름 유래:
"Fine Wine" + "Web" = "FineWeb"

와인 양조처럼:
1. 포도 수확 (Common Crawl)
2. 선별 (필터링)
3. 발효 (처리)
4. 숙성 (중복 제거)
5. 병입 (최종 데이터셋)

최고의 원료 → 최고의 와인
```

### 7.2 물 정화 비유

```
Common Crawl = 강물 (불순물 많음)

정화 과정:
1. 큰 쓰레기 제거 (URL 필터)
2. 모래 필터 (품질 필터)
3. 박테리아 제거 (중복 제거)
4. 미네랄 첨가 (교육성 선별)

결과: 깨끗한 식수 (FineWeb)
```

---

## 8. FineWeb 사용법

### 8.1 데이터셋 로드

```python
from datasets import load_dataset

# 전체 FineWeb (매우 큼!)
# dataset = load_dataset("HuggingFaceFW/fineweb", streaming=True)

# FineWeb-Edu (더 작고 고품질)
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    split="train",
    streaming=True
)

# 샘플 확인
for sample in dataset.take(5):
    print(sample["text"][:500])
    print("---")
```

### 8.2 부분 다운로드

```python
# 특정 snapshot만
dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="CC-MAIN-2024-10",  # 특정 크롤
    split="train"
)
```

---

## 9. 핵심 요약

### 기억해야 할 것들

1. **규모**: 15T 토큰 (영어)
2. **방법**: 5단계 필터링 파이프라인
3. **FineWeb-Edu**: 교육적 콘텐츠 1.3T
4. **핵심**: 품질 필터링이 성능 좌우

### 주요 수치

| 항목 | 값 |
|------|-----|
| 총 토큰 | 15T |
| CC 스냅샷 | 96개 |
| FineWeb-Edu | 1.3T |
| 성능 향상 | +3-5% |

### 파이프라인 요약

$$\text{Raw CC} \xrightarrow{\text{Filter}} \xrightarrow{\text{Quality}} \xrightarrow{\text{Dedup}} \xrightarrow{\text{Edu}} \text{FineWeb}$$

### 실무 팁

- 중복 제거가 가장 효과적
- 교육성 필터로 추가 품질 향상
- 스트리밍으로 대용량 처리
- 언어별 특화 필터 필요

---

## 참고 자료

1. [FineWeb 논문](https://arxiv.org/abs/2406.17557)
2. [FineWeb Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
3. [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
4. [Hugging Face Blog](https://huggingface.co/blog/fineweb)

---

*이전 리뷰: [Phi-1.5](./002_Phi-1.5.md)*
*다음 리뷰: [Cosmopedia](./004_Cosmopedia.md)*
