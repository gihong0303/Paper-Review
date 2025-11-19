# The LLaMA 3 Herd of Models

**논문 발표**: 2024년
**저자**: Meta AI
**소속**: Meta
**논문 링크**: [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
**공식 모델**: [HuggingFace](https://huggingface.co/meta-llama)

---

## 한 줄 요약
> 15T 토큰으로 학습하고 128K 컨텍스트를 지원하는 최대 405B 모델로, GPT-4급 성능을 오픈소스로 공개

---

## 1. LLaMA 3 모델 패밀리

### 1.1 모델 크기

| 모델 | 파라미터 | 학습 토큰 | Context |
|------|----------|-----------|---------|
| 8B | 8B | 15T | 128K |
| 70B | 70B | 15T | 128K |
| **405B** | **405B** | **15T** | **128K** |

### 1.2 vs LLaMA 2

| 항목 | LLaMA 2 | LLaMA 3 |
|------|---------|---------|
| 최대 크기 | 70B | 405B |
| 학습 토큰 | 2T | 15T |
| Context | 4K | 128K |
| Vocab | 32K | 128K |

---

## 2. 아키텍처 개선

### 2.1 확장된 Vocabulary

```
32K → 128K tokens:
- 더 나은 다국어 지원
- 코드 효율성 향상
- 토큰 당 정보량 증가
```

### 2.2 GQA 전 모델 적용

```python
# 모든 모델에 GQA 적용
gqa_config = {
    "8B": {"q_heads": 32, "kv_heads": 8},
    "70B": {"q_heads": 64, "kv_heads": 8},
    "405B": {"q_heads": 128, "kv_heads": 8}
}
```

### 2.3 RoPE θ 증가

```
LLaMA 2: θ = 10,000
LLaMA 3: θ = 500,000

→ 더 긴 컨텍스트 처리 가능
```

---

## 3. 학습 데이터

### 3.1 데이터 구성

```
15T tokens:
- 웹 데이터 (품질 필터링)
- 코드
- 수학
- 다국어

품질 중심:
- 휴리스틱 필터
- Model-based 필터
- 중복 제거
```

### 3.2 데이터 품질 필터링

```python
# 품질 점수 기반 필터링
def filter_data(doc):
    # fastText로 품질 예측
    quality_score = quality_classifier(doc)

    # Roberta 기반 추가 필터
    if quality_score > threshold:
        return True
    return False
```

---

## 4. Long Context (128K)

### 4.1 확장 방법

```
단계적 확장:
1. 8K로 사전 학습
2. 점진적으로 128K까지 확장
3. Synthetic long data 활용
```

### 4.2 Needle in a Haystack

```
128K 컨텍스트 전체에서 정보 검색:
- 정확도: 95%+ (전 구간)
- 위치 무관하게 검색 가능
```

---

## 5. 학습 레시피

### 5.1 Pre-training

```python
training_config = {
    "optimizer": "AdamW",
    "lr": "cosine schedule",
    "batch_size": "점진적 증가",
    "context": "8K → 128K 점진적 확장"
}
```

### 5.2 Post-training

```
SFT → Preference Learning (DPO) → 반복

데이터:
- 인간 annotation
- Synthetic data
- Rejection sampling
```

---

## 6. 실험 결과

### 6.1 벤치마크

| 모델 | MMLU | HumanEval | MATH |
|------|------|-----------|------|
| GPT-4 | 86.4 | 67.0 | 52.9 |
| Claude 3.5 | 88.7 | 92.0 | 71.1 |
| **LLaMA 3 405B** | **88.6** | **89.0** | **73.8** |

### 6.2 다국어 성능

```
MGSM (다국어 수학):
- LLaMA 3 8B: 68.9%
- LLaMA 3 70B: 86.9%
- LLaMA 3 405B: 91.6%

한국어, 일본어, 중국어 등 지원 향상
```

---

## 7. 멀티모달 (Vision)

### 7.1 이미지 인코더

```
Vision encoder 추가:
- Cross-attention으로 연결
- 이미지 이해 능력 확보
```

### 7.2 결과

```
VQA, 이미지 캡셔닝 등:
- GPT-4V에 근접한 성능
```

---

## 8. Tool Use & Safety

### 8.1 도구 사용

```python
# 코드 실행, 검색 등 도구 사용 가능
tools = ["python_interpreter", "web_search", "calculator"]
```

### 8.2 Llama Guard 3

```
안전성 분류기:
- 입력/출력 검사
- 유해 콘텐츠 탐지
- 다국어 지원
```

---

## 9. 핵심 요약

### 기억해야 할 것들

1. **규모**: 405B, 15T 토큰, 128K context
2. **Vocab**: 128K (4배 확대)
3. **성능**: GPT-4급
4. **오픈소스**: 가중치 완전 공개

### 주요 설정

| 항목 | 405B |
|------|------|
| Layers | 126 |
| d_model | 16384 |
| Heads | 128 |
| KV Heads | 8 |
| FFN | 53248 |

### LLaMA 진화

```
LLaMA 1 (2023.02): 65B, 1.4T
LLaMA 2 (2023.07): 70B, 2T
LLaMA 3 (2024.04): 405B, 15T
```

---

## 참고 자료

1. [LLaMA 3 논문](https://arxiv.org/abs/2407.21783)
2. [HuggingFace](https://huggingface.co/meta-llama)
3. [Meta AI Blog](https://ai.meta.com/blog/)

---

*이전 리뷰: [LLaMA 2](./005_LLaMA_2.md)*
*다음 섹션: [Positional Embeddings & Attention](../2_Positional_Attention/)*
