# Prefix-Tuning: Optimizing Continuous Prompts for Generation

**논문 발표**: 2021년 (ACL 2021)
**저자**: Xiang Lisa Li, Percy Liang
**소속**: Stanford University
**논문 링크**: [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)

---

## 한 줄 요약
> 모델 파라미터를 고정하고, 입력 앞에 학습 가능한 연속적 벡터(prefix)를 추가하여 태스크에 적응, 0.1% 파라미터로 Full Fine-tuning 성능 달성

---

## 1. 핵심 아이디어

### 1.1 Discrete vs Continuous Prompt

```
Discrete Prompt (GPT-3 방식):
"Translate English to French: [input]"

Continuous Prompt (Prefix-Tuning):
[P1][P2]...[Pk] + [input]
학습 가능한 벡터들
```

### 1.2 Prefix의 적용

모든 Transformer 레이어의 Key, Value에 prefix 추가:

```
Layer 1: [P_k^1][P_v^1] ... Key, Value
Layer 2: [P_k^2][P_v^2] ... Key, Value
...
Layer L: [P_k^L][P_v^L] ... Key, Value
```

---

## 2. 수학적 표현

$$h_i = \text{Attention}(x_i, [P_K; K], [P_V; V])$$

- $P_K, P_V$: 학습 가능한 prefix
- $K, V$: 원본 key, value

---

## 3. Reparameterization

### 3.1 문제

직접 prefix를 학습하면 불안정

### 3.2 해결

MLP를 통해 간접적으로 생성:

```python
P = MLP(P_embedding)
# 학습 후 MLP 제거, P만 사용
```

---

## 4. 실험 결과

| 방법 | 파라미터 % | Table-to-Text | Summarization |
|------|------------|---------------|---------------|
| Fine-tune | 100% | 기준 | 기준 |
| **Prefix** | **0.1%** | **동등** | **동등** |

---

## 5. 한계점

- 시퀀스 길이 감소 (prefix가 차지)
- 최적화가 LoRA보다 어려움

---

## 참고 자료

1. [Prefix-Tuning 논문](https://arxiv.org/abs/2101.00190)

---

*이전 리뷰: [DoRA](./003_DoRA.md)*
*다음 리뷰: [P-Tuning v2](./005_P-Tuning_v2.md)*
