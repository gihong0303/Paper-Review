# DCLM: DataComp-LM - In Search of the Next Generation of Training Sets for Language Models

**논문 발표**: 2024년 (NeurIPS 2024 Datasets and Benchmarks)
**저자**: Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, 외 100+ 저자
**소속**: Apple, University of Washington, Columbia University, 외 다수
**논문 링크**: [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)
**공식 웹사이트**: [datacomp.ai/dclm](https://www.datacomp.ai/dclm/)

---

## 한 줄 요약
> 240T 토큰의 Common Crawl 코퍼스와 표준화된 평가 파이프라인을 제공하여, 데이터 큐레이션 연구를 가능하게 하고, 간단한 fastText 분류기로 Llama 3와 경쟁하는 오픈 데이터 모델을 달성

---

## 1. 문제 정의

### 1.1 현재 LLM 학습의 불투명성

현재 최고 성능 모델들의 데이터는 비공개:

```
모델별 학습 데이터 공개 여부:
┌─────────────────────────────────────┐
│ 모델        │ 크기   │ 데이터 공개  │
├─────────────┼────────┼──────────────┤
│ GPT-4       │ ~1.7T? │ ❌ 완전 비공개 │
│ Claude      │ ~?     │ ❌ 완전 비공개 │
│ Llama 3     │ 15T    │ ❌ 필터링 비공개│
│ Mistral     │ ~?     │ ❌ 비공개     │
│ Falcon      │ 1T     │ ✅ RefinedWeb │
│ DCLM        │ 4T     │ ✅ 완전 공개  │
└─────────────────────────────────────┘

문제: 데이터 큐레이션 연구가 재현 불가능!
```

### 1.2 기존 오픈 데이터셋의 한계

| 데이터셋 | 크기 | 품질 | 문제점 |
|----------|------|------|--------|
| C4 | 175B | 낮음 | 과도한 필터링 |
| The Pile | 300B | 중간 | 구식, 일부 비공개 |
| RedPajama | 1.2T | 중간 | Llama 학습 데이터 모방 |
| FineWeb | 15T | 높음 | 큐레이션 방법 제한적 |

### 1.3 핵심 질문

> 어떤 데이터 큐레이션 기법이 LLM 성능을 가장 향상시키는가?

이를 답하기 위한 **제어된 실험 환경** 필요!

---

## 2. DCLM 벤치마크 설계

### 2.1 개요

DCLM은 데이터 큐레이션 연구를 위한 **통제된 실험 환경**:

```
DCLM 벤치마크 구성:
┌─────────────────────────────────────────┐
│ 1. Raw Corpus                           │
│    - Common Crawl 240T 토큰             │
│    - 표준화된 전처리                     │
│                                         │
│ 2. Model Training Pipeline              │
│    - OpenLM 프레임워크                   │
│    - 고정된 하이퍼파라미터               │
│    - 412M / 1B / 7B 스케일              │
│                                         │
│ 3. Evaluation Suite                     │
│    - 53개 다운스트림 태스크              │
│    - MMLU, HellaSwag, ARC, 등           │
│                                         │
│ → 데이터만 변수로, 공정한 비교 가능!     │
└─────────────────────────────────────────┘
```

### 2.2 경쟁 트랙

| 트랙 | 설명 | 제약 |
|------|------|------|
| **Filtering** | 데이터 필터링 | 외부 데이터 사용 불가 |
| **Mixing** | 데이터 혼합 비율 | 고정된 소스 사용 |
| **Model-based** | 모델 기반 필터링 | 모든 방법 허용 |

---

## 3. 원시 데이터 처리

### 3.1 Common Crawl 처리 파이프라인

```
원시 데이터 → 정제된 데이터:

1. WARC 파일 다운로드
   └─ Common Crawl의 96개 크롤 (2013-2023)

2. 텍스트 추출
   └─ Resiliparse: 빠르고 정확한 HTML 파싱

3. 언어 필터링
   └─ fastText 기반 영어 감지

4. URL 중복 제거
   └─ Bloom filter: 메모리 효율적

5. 문서 중복 제거
   └─ MinHash + LSH: 대규모 처리 가능

결과: 240T → 약 4T 고품질 토큰
```

### 3.2 텍스트 추출 비교

| 방법 | 속도 | 품질 | 선택 |
|------|------|------|------|
| BeautifulSoup | 느림 | 중간 | ❌ |
| Trafilatura | 중간 | 높음 | ❌ |
| **Resiliparse** | **빠름** | **높음** | ✅ |
| jusText | 중간 | 중간 | ❌ |

Resiliparse가 속도와 품질 모두 우수!

### 3.3 중복 제거

```python
# MinHash 기반 중복 제거
from datasketch import MinHash, MinHashLSH

def deduplicate_documents(documents, threshold=0.8):
    # LSH 인덱스 생성
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    unique_docs = []
    for i, doc in enumerate(documents):
        # MinHash 생성
        mh = MinHash(num_perm=128)
        for word in doc.split():
            mh.update(word.encode('utf8'))

        # 중복 체크
        result = lsh.query(mh)
        if len(result) == 0:
            lsh.insert(i, mh)
            unique_docs.append(doc)

    return unique_docs
```

---

## 4. 핵심 발견: Model-based Filtering

### 4.1 주요 발견

**놀라운 결과**: 간단한 bigram fastText 분류기가 최고 성능!

```
데이터 필터링 방법 비교 (7B 모델, MMLU):

규칙 기반 (품질 지표):
- Perplexity 필터링:     58.2%
- 텍스트 길이 필터링:    56.8%
- 중복 비율 필터링:      57.5%

모델 기반:
- GPT-2 perplexity:      60.1%
- RoBERTa 분류기:        61.3%
- fastText (unigram):    62.8%
- fastText (bigram):     64.0%  ← 최고!

놀라운 점: 간단한 모델이 복잡한 모델보다 우수
```

### 4.2 fastText 분류기 설계

**핵심 아이디어**: 좋은 예시/나쁜 예시 학습

```python
# 학습 데이터 구성
positive_examples = [
    # 고품질 텍스트
    "Wikipedia 문서",
    "학술 논문",
    "책 발췌",
    "고품질 뉴스",
]

negative_examples = [
    # 저품질 텍스트
    "중복 콘텐츠",
    "광고/스팸",
    "기계 생성 텍스트",
    "저품질 웹페이지",
]

# fastText 학습
import fasttext

model = fasttext.train_supervised(
    input="train_data.txt",  # __label__good / __label__bad
    epoch=5,
    lr=0.1,
    wordNgrams=2,  # bigram이 핵심!
    dim=256,
)

# 필터링 적용
def filter_document(doc, threshold=0.5):
    label, prob = model.predict(doc)
    return prob[0] > threshold if label[0] == '__label__good' else False
```

### 4.3 Positive/Negative 예시 선택의 중요성

| Positive 예시 | MMLU | 설명 |
|---------------|------|------|
| Wikipedia | 60.2% | 기본 |
| + OpenWebText2 | 62.1% | Reddit 고품질 |
| + 학술 논문 | 63.5% | 전문 지식 |
| **+ 다양한 고품질** | **64.0%** | 최고 조합 |

### 4.4 왜 간단한 모델이 더 좋은가?

```
이론적 설명:

1. 과적합 방지
   - 복잡한 모델은 예시의 특정 패턴에 과적합
   - 단순 모델은 일반적인 품질 신호 학습

2. 계산 효율
   - 240T 토큰 필터링에 빠른 추론 필수
   - fastText: ~1M 문서/초

3. Bigram의 힘
   - "high quality", "peer review" 같은 품질 패턴
   - Unigram보다 문맥 캡처

4. 인간 판단의 한계
   - 인간은 학습 데이터 품질 판단이 부정확
   - 모델 기반이 더 일관적
```

---

## 5. DCLM-Baseline 데이터셋

### 5.1 최종 파이프라인

```
DCLM-Baseline 생성 과정:

Common Crawl 240T 토큰
         ↓
[Resiliparse 텍스트 추출]
         ↓
[영어 필터링 (fastText)]
         ↓
[URL 중복 제거 (Bloom filter)]
         ↓
[MinHash 중복 제거]
         ↓
[fastText 품질 분류기]
    - Bigram, dim=256
    - Threshold: 0.018 (엄격)
         ↓
DCLM-Baseline 4T 토큰 (~1.7%)
```

### 5.2 데이터 통계

| 지표 | DCLM-Pool | DCLM-Baseline |
|------|-----------|---------------|
| 총 토큰 | 240T | 4T |
| 문서 수 | ~270B | ~3.5B |
| 평균 문서 길이 | 890 토큰 | 1,140 토큰 |
| 필터링 비율 | - | 1.7% |
| 언어 | 영어 | 영어 |

### 5.3 데이터 품질 예시

**포함된 문서 (고품질)**:
```text
The Transformer architecture, introduced in "Attention Is All You Need"
by Vaswani et al. (2017), revolutionized natural language processing
by replacing recurrence with self-attention mechanisms...
```

**제외된 문서 (저품질)**:
```text
Buy now!!! Best deals!!! Click here for FREE!!!
Lorem ipsum dolor sit amet...
[repeated content]...
```

---

## 6. 구현

### 6.1 fastText 분류기 학습

```python
import fasttext
import json

def prepare_training_data(
    positive_sources: list[str],
    negative_sources: list[str],
    output_file: str
):
    """학습 데이터 준비"""
    with open(output_file, 'w') as f:
        # Positive 예시
        for source in positive_sources:
            for doc in load_documents(source):
                # 전처리
                text = preprocess(doc)
                f.write(f"__label__good {text}\n")

        # Negative 예시
        for source in negative_sources:
            for doc in load_documents(source):
                text = preprocess(doc)
                f.write(f"__label__bad {text}\n")

def train_quality_classifier(
    train_file: str,
    output_model: str
):
    """fastText 품질 분류기 학습"""
    model = fasttext.train_supervised(
        input=train_file,
        epoch=5,
        lr=0.1,
        wordNgrams=2,
        dim=256,
        loss='softmax',
        minCount=5,
        thread=32,
    )

    # 모델 저장
    model.save_model(output_model)

    return model


# 사용 예시
positive_sources = [
    "data/wikipedia.txt",
    "data/openwebtext2.txt",
    "data/arxiv.txt",
]

negative_sources = [
    "data/random_cc_sample.txt",  # 무작위 Common Crawl
]

prepare_training_data(
    positive_sources,
    negative_sources,
    "train_data.txt"
)

model = train_quality_classifier(
    "train_data.txt",
    "quality_classifier.bin"
)
```

### 6.2 대규모 필터링 파이프라인

```python
import fasttext
from multiprocessing import Pool
import pyarrow.parquet as pq

class DCLMFilter:
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model = fasttext.load_model(model_path)
        self.threshold = threshold

    def filter_document(self, text: str) -> bool:
        """단일 문서 필터링"""
        # 전처리
        text = ' '.join(text.split()[:512])  # 최대 512 단어

        # 예측
        labels, probs = self.model.predict(text)

        # 품질 판단
        if labels[0] == '__label__good':
            return probs[0] >= self.threshold
        else:
            return probs[0] < (1 - self.threshold)

    def filter_batch(self, documents: list[str]) -> list[str]:
        """배치 필터링"""
        return [doc for doc in documents if self.filter_document(doc)]


def process_shard(args):
    """단일 샤드 처리"""
    shard_path, model_path, threshold, output_path = args

    filter_model = DCLMFilter(model_path, threshold)

    # 데이터 로드
    table = pq.read_table(shard_path)
    documents = table['text'].to_pylist()

    # 필터링
    filtered = filter_model.filter_batch(documents)

    # 저장
    pq.write_table(
        pa.table({'text': filtered}),
        output_path
    )

    return len(documents), len(filtered)


def filter_common_crawl(
    input_dir: str,
    output_dir: str,
    model_path: str,
    threshold: float = 0.018,
    num_workers: int = 64
):
    """전체 Common Crawl 필터링"""
    import glob
    import os

    shard_paths = glob.glob(f"{input_dir}/*.parquet")

    args_list = [
        (
            shard,
            model_path,
            threshold,
            f"{output_dir}/{os.path.basename(shard)}"
        )
        for shard in shard_paths
    ]

    # 병렬 처리
    with Pool(num_workers) as pool:
        results = pool.map(process_shard, args_list)

    # 통계 출력
    total_input = sum(r[0] for r in results)
    total_output = sum(r[1] for r in results)
    print(f"Filtered: {total_input} -> {total_output}")
    print(f"Keep ratio: {total_output/total_input:.2%}")
```

### 6.3 DCLM 데이터 사용

```python
from datasets import load_dataset

# HuggingFace에서 DCLM-Baseline 로드
dataset = load_dataset(
    "mlfoundations/dclm-baseline-1.0",
    split="train",
    streaming=True,  # 스트리밍 모드 (대용량)
)

# 샘플 확인
for i, sample in enumerate(dataset):
    print(f"Text: {sample['text'][:200]}...")
    if i >= 5:
        break

# 학습에 사용
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)
```

---

## 7. 쉬운 예시로 이해하기

### 7.1 도서관 사서 비유

**문제**: 100만 권의 책 중 좋은 책만 선별

```
방법 1: 규칙 기반
- "페이지가 100장 이상이면 좋은 책"
- "글자 크기가 12pt이면 좋은 책"
→ 표면적 특징만 봄, 내용 무시

방법 2: 전문가 (복잡한 모델)
- 각 책을 전문가가 꼼꼼히 읽음
- 매우 정확하지만 너무 느림
→ 100만 권 처리 불가능

방법 3: 빠른 스크리닝 (fastText)
- "참고문헌이 있는지"
- "학술적 용어가 있는지"
→ 빠르게 대량 처리, 충분히 정확
```

### 7.2 물 정수 비유

```
원수 (Common Crawl 240T):
- 좋은 물 + 불순물 + 독소

단계별 정수:
1. 거친 필터 (텍스트 추출)
   - 큰 이물질 제거
   - HTML 태그, 스크립트 등

2. 모래 필터 (중복 제거)
   - 중간 입자 제거
   - 같은 문서 반복 제거

3. 활성탄 필터 (품질 분류)
   - 미세 불순물 제거
   - 저품질 콘텐츠 제거

정수된 물 (DCLM-Baseline 4T):
- 깨끗한 고품질 데이터만 남음
```

### 7.3 숫자로 보는 선별

```
DCLM 데이터 정제 과정:

원시 Common Crawl
    │
    ↓ 텍스트 추출
    │ (HTML → Text)
    │
   240T 토큰
    │
    ↓ 언어 필터링
    │ (영어만)
    │
   120T 토큰 (50%)
    │
    ↓ 중복 제거
    │ (URL + 내용)
    │
   40T 토큰 (33%)
    │
    ↓ 품질 필터링
    │ (fastText 분류기)
    │
   4T 토큰 (10%)
    │
    ↓ 최종 결과
    │
DCLM-Baseline

총 필터링 비율: 4T / 240T = 1.7%
→ 상위 1.7%의 고품질 데이터만 사용
```

---

## 8. 실험 결과

### 8.1 DCLM-7B 성능

| 모델 | 학습 데이터 | MMLU | Compute |
|------|-------------|------|---------|
| MAP-Neo 7B | 4.5T | 57.4% | 기준 |
| Falcon 7B | 1.5T | 55.4% | 0.3× |
| Llama 2 7B | 2T | 45.9% | 0.3× |
| Mistral 7B | ~? | 63.0% | ~? |
| Llama 3 8B | 15T | 66.0% | 6.6× |
| **DCLM-7B** | **2.6T** | **64.0%** | **1.0×** |

**DCLM-7B는 Llama 3 8B의 1/6 계산량으로 근접한 성능!**

### 8.2 53개 태스크 평균 성능

```
모델별 평균 성능 (53 tasks):

Llama 3 8B:    ████████████████████  62.1%
DCLM-7B:       ███████████████████░  60.8%
Mistral 7B:    ██████████████████░░  58.4%
MAP-Neo 7B:    █████████████████░░░  56.2%
Falcon 7B:     ████████████████░░░░  54.1%

                0%       25%      50%      75%
```

### 8.3 데이터 품질 vs 양

| 설정 | 토큰 수 | MMLU | 효율 |
|------|---------|------|------|
| 전체 CC | 240T | 48.2% | 0.20 |
| 규칙 기반 필터 | 40T | 56.8% | 1.42 |
| Perplexity 필터 | 20T | 58.2% | 2.91 |
| **DCLM-Baseline** | **4T** | **64.0%** | **16.0** |

**품질이 양보다 훨씬 중요!**

### 8.4 Ablation Study

| 변형 | MMLU | 설명 |
|------|------|------|
| DCLM-Baseline | 64.0% | 기본 설정 |
| - 중복 제거 | 61.2% | 중복 남김 |
| - 품질 필터 | 52.4% | 필터링 없음 |
| Unigram 분류기 | 62.8% | bigram → unigram |
| 높은 threshold | 65.1% | 더 엄격한 필터링 |
| 낮은 threshold | 60.3% | 느슨한 필터링 |

### 8.5 Scaling Law

```
모델 크기별 성능 (MMLU):

                412M    1B      7B
───────────────────────────────────
C4              35.2%   42.1%   52.3%
RedPajama       36.8%   44.5%   56.1%
DCLM-Baseline   40.1%   48.8%   64.0%

DCLM의 개선폭이 스케일에서 유지/증가!
```

---

## 9. 한계점 및 후속 연구

### 9.1 현재 한계점

1. **영어 전용**: 다국어 지원 없음
   - 영어 외 언어 필터링 연구 필요

2. **단일 분류기**: 도메인별 최적화 없음
   - 수학, 코드 등 특수 도메인 별도 처리 필요

3. **정적 필터링**: 학습 중 적응 없음
   - Curriculum learning과 결합 가능

4. **Human evaluation 부족**: 자동 메트릭만 사용
   - 실제 사용자 선호도 평가 필요

### 9.2 후속 연구 방향

1. **다국어 DCLM**: 100+ 언어 지원
2. **도메인 특화**: 코드, 수학, 과학 분야 별도 파이프라인
3. **동적 필터링**: 학습 중 데이터 선택 조정
4. **Synthetic data 통합**: 합성 데이터와 결합

### 9.3 영향

- **오픈 데이터 연구 활성화**: 재현 가능한 벤치마크
- **데이터 큐레이션 중요성 인식**: 데이터 > 모델 크기
- **기업 vs 학계 격차 축소**: 고품질 오픈 데이터 제공

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **핵심 발견**: 간단한 fastText bigram 분류기가 최고 성능
2. **데이터 품질**: 양보다 품질이 훨씬 중요 (1.7% 필터링)
3. **DCLM-Baseline**: Llama 3와 경쟁하는 완전 오픈 데이터
4. **벤치마크 제공**: 통제된 데이터 큐레이션 실험 환경

### 핵심 수치

| 지표 | 값 |
|------|-----|
| 원시 데이터 | 240T 토큰 |
| 필터링 후 | 4T 토큰 |
| 필터링 비율 | 1.7% |
| MMLU (7B) | 64.0% |
| Compute 효율 | Llama 3의 1/6 |

### 실무 체크리스트

```python
# 1. 데이터 로드
from datasets import load_dataset
dataset = load_dataset("mlfoundations/dclm-baseline-1.0")

# 2. 자체 분류기 학습 (커스텀 필터링)
import fasttext
model = fasttext.train_supervised(
    input="train_data.txt",
    wordNgrams=2,  # bigram 필수!
)

# 3. 필터링 적용
def filter_docs(docs, threshold=0.5):
    return [d for d in docs
            if model.predict(d)[1][0] > threshold]

# 4. 학습 실행
# OpenLM 또는 기타 프레임워크 사용
```

---

## 참고 자료

1. [DCLM 논문](https://arxiv.org/abs/2406.11794)
2. [공식 웹사이트](https://www.datacomp.ai/dclm/)
3. [GitHub 저장소](https://github.com/mlfoundations/dclm)
4. [HuggingFace 데이터셋](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)
5. [DCLM-7B 모델](https://huggingface.co/apple/DCLM-7B)

---

*이전 리뷰: [Self-Instruct](./005_Self-Instruct.md)*
*다음 리뷰: [Magicoder](./007_Magicoder.md)*
