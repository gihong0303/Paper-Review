# SentencePiece: A simple and language independent subword tokenizer and detokenizer

**논문 발표**: 2018년 (EMNLP 2018)
**저자**: Taku Kudo, John Richardson
**소속**: Google
**논문 링크**: [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
**공식 구현**: [GitHub](https://github.com/google/sentencepiece)

---

## 한 줄 요약
> 전처리 없이 raw text에서 직접 subword 토큰화를 수행하는 언어 독립적 라이브러리로, BPE와 Unigram 알고리즘을 모두 지원

---

## 1. 기존 토크나이저의 문제

### 1.1 언어 의존적 전처리

```
기존 방식:
raw text → 토크나이징 → subword 분할
             ↑
    언어별 다른 처리 필요

영어: space로 분리
중국어: 글자 단위
일본어: MeCab 등 형태소 분석
한국어: 형태소 분석기
```

### 1.2 가역성 문제

```
Detokenization 문제:
["Hello", ",", "world"] → "Hello, world" (어려움)

공백 정보 손실:
- 원본: "New York"
- 토큰: ["New", "York"]
- 복원: "NewYork"? "New York"?
```

---

## 2. SentencePiece의 해결책

### 2.1 핵심 아이디어

```
1. 공백을 특수 문자(▁)로 치환
2. Raw text에서 직접 학습
3. 언어 독립적 처리
4. 완전한 가역성
```

### 2.2 공백 처리

```python
# 공백을 ▁(U+2581)로 치환
original = "Hello World"
processed = "▁Hello▁World"

# 토큰화
tokens = ["▁Hello", "▁Wor", "ld"]

# 완벽한 복원
detokenized = "".join(tokens).replace("▁", " ")
# "Hello World"
```

---

## 3. 지원 알고리즘

### 3.1 BPE (Byte Pair Encoding)

```python
# BPE 모델 학습
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='bpe',
    vocab_size=32000,
    model_type='bpe'  # BPE 선택
)
```

### 3.2 Unigram

```python
# Unigram 모델 학습
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='unigram',
    vocab_size=32000,
    model_type='unigram'  # Unigram 선택
)
```

### 3.3 BPE vs Unigram

| 특성 | BPE | Unigram |
|------|-----|---------|
| 방향 | Bottom-up | Top-down |
| 분할 | 결정적 | 확률적 |
| 속도 | 빠름 | 느림 |
| 품질 | 좋음 | 더 좋음 |

---

## 4. Unigram 알고리즘 상세

### 4.1 개념

```
확률 모델 기반:
- 각 subword에 확률 부여
- 최적 분할 = 확률 최대화

예: "unrelated"
분할 1: ["un", "related"] → P = 0.4 × 0.3 = 0.12
분할 2: ["unre", "lated"] → P = 0.1 × 0.1 = 0.01

→ 분할 1 선택
```

### 4.2 수식

$$P(x) = \prod_{i=1}^{n} p(x_i)$$

여기서 $x = (x_1, ..., x_n)$은 subword 시퀀스

최적 분할:
$$x^* = \arg\max_x P(x)$$

### 4.3 학습 과정

```
1. 큰 vocabulary로 시작 (예: 100만)
2. EM 알고리즘으로 확률 추정
3. 낮은 확률 subword 제거
4. 목표 크기까지 반복
```

---

## 5. 코드 사용법

### 5.1 모델 학습

```python
import sentencepiece as spm

# 학습
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='mymodel',
    vocab_size=32000,
    character_coverage=0.9995,  # 문자 커버리지
    model_type='unigram',
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)
```

### 5.2 토큰화

```python
# 모델 로드
sp = spm.SentencePieceProcessor()
sp.load('mymodel.model')

# 인코딩
text = "Hello, World!"
pieces = sp.encode_as_pieces(text)
# ['▁Hello', ',', '▁World', '!']

ids = sp.encode_as_ids(text)
# [1234, 45, 6789, 10]

# 디코딩
decoded = sp.decode_pieces(pieces)
# "Hello, World!"

decoded = sp.decode_ids(ids)
# "Hello, World!"
```

### 5.3 특수 옵션

```python
# BPE dropout (regularization)
sp.encode("Hello", enable_sampling=True, alpha=0.1)

# N-best 샘플링
sp.nbest_encode_as_pieces("Hello", nbest_size=5)
```

---

## 6. 주요 특징

### 6.1 언어 독립성

```
동일한 코드로 모든 언어 처리:

영어: "Hello World" → ["▁Hello", "▁World"]
한국어: "안녕하세요" → ["▁안녕", "하세요"]
일본어: "こんにちは" → ["▁こんに", "ちは"]
중국어: "你好世界" → ["▁你好", "世界"]
```

### 6.2 Subword Regularization

```python
# 학습 시 다양한 분할 사용 (Unigram)
# 동일 문장도 매번 다르게 토큰화
# → 일반화 성능 향상

for epoch in range(num_epochs):
    tokens = sp.encode(text, enable_sampling=True)
    # 매번 다른 분할 가능
```

### 6.3 Character Coverage

```python
# 희귀 문자 처리
character_coverage = 0.9995

# 전체 문자의 99.95%를 커버
# 나머지는 <unk> 또는 바이트로 처리
```

---

## 7. LLM에서의 사용

### 7.1 사용 모델

| 모델 | Tokenizer | Vocab |
|------|-----------|-------|
| LLaMA | SentencePiece (BPE) | 32K |
| T5 | SentencePiece (Unigram) | 32K |
| ALBERT | SentencePiece (Unigram) | 30K |
| mBART | SentencePiece (BPE) | 250K |

### 7.2 LLaMA 토크나이저 예시

```python
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")

text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
# ['▁Hello', ',', '▁how', '▁are', '▁you', '?']
```

---

## 8. 한국어 처리

### 8.1 한국어 특화 학습

```python
# 한국어 코퍼스로 학습
spm.SentencePieceTrainer.train(
    input='korean_corpus.txt',
    model_prefix='ko_spm',
    vocab_size=50000,
    character_coverage=0.9999,  # 한글 커버리지 높게
    user_defined_symbols=['<sep>', '<cls>'],
    model_type='unigram'
)
```

### 8.2 형태소 분석과 결합

```python
from konlpy.tag import Mecab

mecab = Mecab()
sp = spm.SentencePieceProcessor()
sp.load('ko_spm.model')

def tokenize_korean(text):
    # 형태소 분석 후 SentencePiece
    morphs = mecab.morphs(text)
    return sp.encode_as_pieces(' '.join(morphs))

text = "자연어처리를 공부합니다"
tokens = tokenize_korean(text)
# 더 의미 있는 분할
```

---

## 9. 쉬운 예시

### 9.1 공백 문자 비유

```
기존 방식 = 책에서 띄어쓰기 제거
"Hello World" → "HelloWorld"
→ 어디서 단어가 나뉘는지 모름!

SentencePiece = 띄어쓰기에 마커 추가
"Hello World" → "▁Hello▁World"
→ 완벽하게 복원 가능!
```

### 9.2 레시피 비유

```
기존: 언어마다 다른 레시피
- 영어: 레시피 A
- 중국어: 레시피 B
- 한국어: 레시피 C

SentencePiece: 하나의 만능 레시피
- 어떤 재료(언어)든 동일하게 처리
```

---

## 10. HuggingFace와 통합

### 10.1 Transformers에서 사용

```python
from transformers import AutoTokenizer

# SentencePiece 기반 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

# 토큰화
text = "Hello, World!"
encoded = tokenizer(text)
print(tokenizer.convert_ids_to_tokens(encoded['input_ids']))
```

### 10.2 직접 변환

```python
from transformers import PreTrainedTokenizerFast

# SentencePiece를 HuggingFace 형식으로 변환
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="mymodel.model",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>"
)
```

---

## 11. 핵심 요약

### 기억해야 할 것들

1. **핵심**: Raw text에서 직접 토큰화
2. **특징**: 공백을 ▁로 치환 → 완벽한 가역성
3. **알고리즘**: BPE, Unigram 지원
4. **장점**: 언어 독립적, 전처리 불필요

### 주요 기능

| 기능 | 설명 |
|------|------|
| 언어 독립 | 어떤 언어도 동일하게 처리 |
| 가역성 | 토큰 → 원본 완벽 복원 |
| Subword Reg. | 학습 시 다양한 분할 |
| 직접 학습 | Raw text에서 바로 학습 |

### 실무 팁

- Vocab 크기: 32K ~ 64K
- Character coverage: 0.9995+
- 한국어: 별도 코퍼스로 학습
- Unigram: 품질 우선 시 선택

### 명령어 요약

```bash
# CLI로 학습
spm_train --input=corpus.txt --model_prefix=m --vocab_size=32000

# 인코딩
spm_encode --model=m.model --output_format=piece < input.txt
```

---

## 참고 자료

1. [SentencePiece 논문](https://arxiv.org/abs/1808.06226)
2. [GitHub](https://github.com/google/sentencepiece)
3. [Google AI Blog](https://ai.googleblog.com/)

---

*이전 리뷰: [BPE](./001_BPE.md)*
*다음 리뷰: [XGLM](./003_XGLM.md)*
