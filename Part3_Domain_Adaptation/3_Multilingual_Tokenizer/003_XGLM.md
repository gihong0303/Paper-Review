# XGLM: Few-shot Learning with Multilingual Language Models

**논문 발표**: 2021년 (EMNLP 2022)
**저자**: Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, et al.
**소속**: Meta AI
**논문 링크**: [arXiv:2112.10668](https://arxiv.org/abs/2112.10668)
**공식 모델**: [HuggingFace](https://huggingface.co/facebook/xglm-7.5B)

---

## 한 줄 요약
> 134개 언어로 학습된 최대 7.5B 파라미터의 다국어 GPT 모델로, 저자원 언어에서도 few-shot learning 능력을 입증

---

## 1. 배경: 다국어 LLM의 필요성

### 1.1 기존 상황

```
GPT-3 (2020):
- 뛰어난 few-shot 능력
- 하지만 영어 중심 (93%)
- 다른 언어 성능 제한적

다국어 모델 필요:
- 전 세계 7000+ 언어
- 대부분 저자원(low-resource)
- 영어만으로는 불충분
```

### 1.2 XGLM의 목표

```
GPT-3의 few-shot 능력을 다국어로!

1. 더 균형 잡힌 언어 분포
2. 저자원 언어 지원
3. Cross-lingual transfer
```

---

## 2. 모델 구성

### 2.1 모델 크기

| 모델 | 파라미터 | 레이어 | Hidden | Heads |
|------|----------|--------|--------|-------|
| XGLM-564M | 564M | 24 | 1024 | 16 |
| XGLM-1.7B | 1.7B | 24 | 2048 | 16 |
| XGLM-2.9B | 2.9B | 48 | 2048 | 32 |
| XGLM-4.5B | 4.5B | 48 | 2560 | 32 |
| **XGLM-7.5B** | **7.5B** | **32** | **4096** | **32** |

### 2.2 학습 데이터

```
CC100 + CC-Net (500B tokens)

언어별 분포:
- 영어: ~17% (상한선)
- 기타 고자원: 10-17%
- 저자원: 가능한 최대 비율
```

---

## 3. 데이터 샘플링 전략

### 3.1 문제: 언어 불균형

```
자연 분포:
- 영어: 50%+
- 중국어: 10%
- 한국어: 1%
- 줄루어: 0.001%

→ 저자원 언어 학습 불가능
```

### 3.2 해결: Temperature Sampling

$$p_l \propto n_l^\alpha$$

여기서:
- $p_l$: 언어 $l$의 샘플링 확률
- $n_l$: 언어 $l$의 토큰 수
- $\alpha$: temperature (0.3 사용)

```python
def temperature_sampling(language_counts, alpha=0.3):
    # 각 언어 토큰 수에 alpha 제곱
    weights = {lang: count ** alpha
               for lang, count in language_counts.items()}

    # 정규화
    total = sum(weights.values())
    probs = {lang: w / total for lang, w in weights.items()}

    return probs
```

### 3.3 결과

```
α = 0.3 사용 시:
- 영어: 50% → 17%
- 한국어: 1% → 5%
- 저자원 언어: 크게 증가
```

---

## 4. 토크나이저

### 4.1 다국어 SentencePiece

```python
# 256K vocabulary
# 모든 134개 언어 커버

tokenizer_config = {
    "vocab_size": 256000,
    "character_coverage": 0.9999,
    "model_type": "unigram"
}
```

### 4.2 언어별 토큰 효율

| 언어 | 토큰/단어 비율 |
|------|----------------|
| 영어 | 1.2 |
| 한국어 | 2.1 |
| 일본어 | 1.8 |
| 힌디어 | 2.5 |

---

## 5. 실험 결과

### 5.1 Few-shot 성능

| 태스크 | XGLM-7.5B | mGPT | GPT-3 (영어) |
|--------|-----------|------|--------------|
| XNLI (평균) | 55.3 | 42.1 | 47.2* |
| XStoryCloze | 66.8 | 58.4 | - |
| XCOPA | 61.5 | 53.2 | - |

*영어만 테스트

### 5.2 언어별 성능

XNLI Zero-shot:

| 언어 | 정확도 |
|------|--------|
| 영어 | 68.2 |
| 프랑스어 | 62.1 |
| 한국어 | 53.8 |
| 스와힐리어 | 48.5 |

### 5.3 Cross-lingual Transfer

```
영어로 few-shot → 다른 언어 테스트

XNLI (영어 예시 → 타겟 언어):
- 프랑스어: +5.2%
- 독일어: +4.8%
- 한국어: +3.1%

영어 예시로도 다국어 성능 향상!
```

---

## 6. 핵심 발견

### 6.1 스케일링 효과

```
모델 크기 증가에 따른 few-shot 개선:
- 564M → 1.7B: +8%
- 1.7B → 7.5B: +12%

저자원 언어에서 더 큰 개선
```

### 6.2 언어 유사성 효과

```
유사 언어 간 transfer 더 좋음:
- 스페인어 ↔ 포르투갈어: 높은 transfer
- 영어 ↔ 중국어: 낮은 transfer
```

### 6.3 In-context Learning

```
예시 언어의 영향:
1. 타겟 언어 예시 > 영어 예시
2. 하지만 영어 예시도 도움 됨
3. 예시 수 증가 → 성능 향상
```

---

## 7. 사용법

### 7.1 기본 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/xglm-7.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 다국어 생성
prompts = [
    "Hello, how are you?",
    "안녕하세요, 어떻게 지내세요?",
    "你好，你好吗？"
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    print(tokenizer.decode(outputs[0]))
```

### 7.2 Few-shot 예시

```python
# 한국어 감정 분류
prompt = """
문장: 이 영화 정말 재미있어요!
감정: 긍정

문장: 음식이 너무 짜서 못 먹겠어요.
감정: 부정

문장: 오늘 날씨가 좋네요.
감정:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 5)
```

---

## 8. 한국어 성능

### 8.1 한국어 테스트

```
KLUE 벤치마크 (Few-shot):
- 감정 분석: 58%
- 자연어 추론: 52%
- 개체명 인식: 45%

개선 여지 있음:
- 한국어 전용 모델 대비 낮음
- 하지만 추가 학습 없이도 동작
```

### 8.2 한국어 개선 방향

```
1. 한국어 데이터로 추가 학습 (CPT)
2. 한국어 instruction tuning
3. 토크나이저 확장
```

---

## 9. 쉬운 예시

### 9.1 다국어 비서 비유

```
단일 언어 모델 = 영어만 아는 비서
- 영어 업무: 완벽
- 한국어 업무: 불가능

XGLM = 134개 언어 구사하는 비서
- 모든 언어 업무 가능
- 각 언어 전문가만큼은 아니지만
- 기본적인 의사소통 가능
```

### 9.2 언어 학습 비유

```
Temperature Sampling = 균형 있는 언어 학습

자연스러운 학습:
- 영어: 매일 10시간
- 한국어: 주 1시간
→ 영어만 잘함

균형 학습:
- 영어: 매일 3시간
- 한국어: 매일 2시간
→ 둘 다 적절히
```

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **규모**: 7.5B 파라미터, 134개 언어
2. **방법**: Temperature sampling (α=0.3)
3. **특징**: Few-shot 다국어 능력
4. **결과**: 저자원 언어에서도 동작

### 주요 수치

| 항목 | 값 |
|------|-----|
| 최대 파라미터 | 7.5B |
| 언어 수 | 134 |
| 학습 토큰 | 500B |
| Vocab 크기 | 256K |

### 샘플링 공식

$$p_l \propto n_l^{0.3}$$

저자원 언어 비율 증가!

### 실무 팁

- 저자원 언어: α 낮추기 (0.2~0.3)
- 큰 모델일수록 cross-lingual 좋음
- 타겟 언어 예시가 가장 효과적
- 한국어: 추가 CPT 권장

---

## 참고 자료

1. [XGLM 논문](https://arxiv.org/abs/2112.10668)
2. [HuggingFace](https://huggingface.co/facebook/xglm-7.5B)
3. [Meta AI Blog](https://ai.facebook.com/blog/)

---

*이전 리뷰: [SentencePiece](./002_SentencePiece.md)*
*다음 리뷰: [BLOOM](./004_BLOOM.md)*
