# LLaMA 2: Open Foundation and Fine-Tuned Chat Models

**논문 발표**: 2023년
**저자**: Hugo Touvron, Louis Martin, Kevin Stone, et al.
**소속**: Meta AI
**논문 링크**: [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
**공식 모델**: [HuggingFace](https://huggingface.co/meta-llama)

---

## 한 줄 요약
> 40% 더 많은 데이터로 학습하고 RLHF로 정렬한 오픈소스 LLM으로, GPT-3.5에 근접한 성능을 공개 모델로 달성

---

## 1. LLaMA 1 → LLaMA 2

### 1.1 주요 개선

| 항목 | LLaMA 1 | LLaMA 2 |
|------|---------|---------|
| 학습 토큰 | 1.0T | 2.0T |
| Context | 2048 | 4096 |
| GQA | 없음 | 34B, 70B에 적용 |
| Chat 모델 | 없음 | RLHF 적용 |

### 1.2 모델 크기

| 모델 | 파라미터 | Context | GQA |
|------|----------|---------|-----|
| 7B | 6.7B | 4096 | No |
| 13B | 13B | 4096 | No |
| 34B | 34B | 4096 | Yes |
| **70B** | **70B** | **4096** | **Yes** |

---

## 2. 아키텍처

### 2.1 기본 구성

```python
architecture = {
    "type": "Decoder-only Transformer",
    "normalization": "RMSNorm (pre-norm)",
    "activation": "SwiGLU",
    "position": "RoPE",
    "attention": "GQA (34B, 70B)"
}
```

### 2.2 Grouped-Query Attention (GQA)

```
MHA: Q heads = K heads = V heads
MQA: Q heads > K heads = V heads = 1
GQA: Q heads > K heads = V heads > 1

LLaMA 2 70B:
- Query heads: 64
- KV heads: 8
- 8배 KV 캐시 절약
```

---

## 3. 학습

### 3.1 Pre-training

```
데이터: 2T tokens
- 공개 데이터만 사용
- 개인정보 필터링

설정:
- AdamW (β1=0.9, β2=0.95)
- Cosine LR schedule
- 2000 step warmup
```

### 3.2 안전성 필터링

```
학습 데이터 필터링:
1. 개인정보 제거
2. 유해 콘텐츠 제거
3. 품질 필터링
```

---

## 4. LLaMA 2-Chat: RLHF

### 4.1 학습 파이프라인

```
Pre-training
     ↓
Supervised Fine-tuning (SFT)
     ↓
RLHF (Rejection Sampling + PPO)
     ↓
LLaMA 2-Chat
```

### 4.2 RLHF 상세

```python
# Reward Model 학습
# 인간 선호도 데이터 사용

# PPO 최적화
loss = -reward + β * KL(π, π_ref)
```

### 4.3 Ghost Attention (GAtt)

```
시스템 프롬프트를 대화 전체에 걸쳐 유지:

System: "You are a helpful assistant"
User: "Hello"
Assistant: [GAtt로 시스템 프롬프트 참조]
User: "What's 2+2?"
Assistant: [여전히 시스템 프롬프트 참조]
```

---

## 5. 실험 결과

### 5.1 Base Model

| 모델 | MMLU | TriviaQA | NQ |
|------|------|----------|-----|
| LLaMA 1 65B | 63.4 | 70.7 | 27.8 |
| **LLaMA 2 70B** | **68.9** | **80.7** | **33.0** |
| GPT-3.5 | 70.0 | - | - |

### 5.2 Chat Model

| 모델 | Helpfulness | Safety |
|------|-------------|--------|
| ChatGPT | 기준 | 기준 |
| LLaMA 2-Chat 70B | -4.9% | +8.4% |

---

## 6. 안전성

### 6.1 Red Teaming

```
전문가 350명으로 안전성 테스트:
- 다양한 공격 시도
- 취약점 발견 및 수정
- 반복적 개선
```

### 6.2 Safety RLHF

```
두 가지 Reward Model:
1. Helpfulness RM
2. Safety RM

Safety RM 점수가 임계값 이하면 거부
```

---

## 7. 핵심 요약

### 기억해야 할 것들

1. **데이터**: 2T 토큰 (LLaMA 1의 2배)
2. **Context**: 4096 (2배)
3. **GQA**: KV 캐시 효율화
4. **Chat**: RLHF + Ghost Attention

### 주요 설정

| 항목 | 값 |
|------|-----|
| Vocab | 32K |
| RoPE θ | 10000 |
| LR | 3e-4 (7B) ~ 1.5e-4 (70B) |

---

## 참고 자료

1. [LLaMA 2 논문](https://arxiv.org/abs/2307.09288)
2. [HuggingFace](https://huggingface.co/meta-llama)

---

*이전 리뷰: [Chinchilla](./004_Chinchilla.md)*
*다음 리뷰: [LLaMA 3](./006_LLaMA_3.md)*
