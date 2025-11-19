# Mixtral of Experts

**논문 발표**: 2024년
**저자**: Albert Q. Jiang, Alexandre Sablayrolles, et al.
**소속**: Mistral AI
**논문 링크**: [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
**공식 모델**: [HuggingFace](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)

---

## 한 줄 요약
> 8개의 7B expert 중 2개를 선택하는 Sparse MoE로, 46B 파라미터지만 12B급 연산으로 LLaMA 2 70B를 능가

---

## 1. Mixtral 구조

### 1.1 기본 구성

```
총 파라미터: 46.7B
활성 파라미터: 12.9B (추론 시)

8 experts × 7B FFN
Top-2 routing
```

### 1.2 아키텍처

```
Mistral 7B 기반:
- Sliding Window Attention
- GQA
- RoPE

+ MoE Layer (FFN 대체)
```

---

## 2. Sparse MoE Layer

### 2.1 Top-2 Routing

```python
def mixtral_moe(x, experts, router):
    # Router probabilities
    router_logits = router(x)  # [batch, seq, 8]
    weights = F.softmax(router_logits, dim=-1)

    # Top-2 selection
    top_weights, top_indices = weights.topk(2, dim=-1)
    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

    # Expert computation
    output = torch.zeros_like(x)
    for i, expert in enumerate(experts):
        # 이 expert가 선택된 토큰들
        mask = (top_indices == i).any(dim=-1)
        if mask.any():
            expert_out = expert(x[mask])
            # 가중합
            weight = top_weights[mask, (top_indices[mask] == i).nonzero()[:, 1]]
            output[mask] += weight.unsqueeze(-1) * expert_out

    return output
```

### 2.2 수식

$$y = \sum_{i \in \text{Top-2}} g_i(x) \cdot E_i(x)$$

---

## 3. 실험 결과

### 3.1 벤치마크

| 모델 | 활성 파라미터 | MMLU | GSM8K |
|------|---------------|------|-------|
| LLaMA 2 70B | 70B | 68.9 | 56.8 |
| **Mixtral 8x7B** | **12.9B** | **70.6** | **74.4** |

**5배 적은 연산으로 더 좋은 성능!**

### 3.2 추론 속도

```
LLaMA 2 70B: 1x (기준)
Mixtral 8x7B: 6x 빠름

같은 GPU 메모리에서
더 높은 처리량
```

---

## 4. Expert 분석

### 4.1 Expert Specialization

```
Expert별 토큰 분포:
- Expert 1: 주로 코드
- Expert 2: 주로 수학
- Expert 3: 주로 자연어
- ...

자연스럽게 전문화됨
```

### 4.2 Routing 패턴

```
동일 문장 내에서도:
- 단어마다 다른 expert 선택
- 문맥에 따라 동적 라우팅
```

---

## 5. 장점과 한계

### 5.1 장점

```
1. 높은 성능/연산 효율
2. 빠른 추론
3. 전문화된 experts
```

### 5.2 한계

```
1. 메모리: 전체 46B 필요
2. 통신: 분산 시 오버헤드
3. 배치 효율: Load balancing
```

---

## 6. 핵심 요약

### 기억해야 할 것들

1. **구조**: 8 experts, Top-2 선택
2. **효율**: 46B 파라미터, 12.9B 활성
3. **성능**: LLaMA 2 70B 능가
4. **의의**: MoE의 상용화 성공

### Mixtral 설정

| 항목 | 값 |
|------|-----|
| Experts | 8 |
| Expert size | 7B (FFN) |
| Top-k | 2 |
| 총 파라미터 | 46.7B |
| 활성 파라미터 | 12.9B |

---

## 참고 자료

1. [Mixtral 논문](https://arxiv.org/abs/2401.04088)

---

*이전 리뷰: [Switch Transformer](./001_Switch_Transformer.md)*
*다음 리뷰: [DeepSeek-V2](./003_DeepSeek-V2.md)*
