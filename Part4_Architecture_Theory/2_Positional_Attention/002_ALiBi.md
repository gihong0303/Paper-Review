# ALiBi: Attention with Linear Biases

**논문 발표**: 2021년 (ICLR 2022)
**저자**: Ofir Press, Noah A. Smith, Mike Lewis
**소속**: University of Washington, Meta AI
**논문 링크**: [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)

---

## 한 줄 요약
> Position embedding 대신 attention score에 거리 기반 선형 bias를 추가하여, 짧은 시퀀스로 학습하고 긴 시퀀스에서 추론 가능

---

## 1. 핵심 아이디어

### 1.1 방법

$$\text{softmax}(q_i K^T + m \cdot [-(i-1), ..., -1, 0])$$

```
위치 임베딩 추가 대신
Attention score에서 거리만큼 빼기

멀수록 → 더 많이 빼기 → attention 낮음
```

### 1.2 시각화

```
Position: 1  2  3  4  5
Query 5:  -4 -3 -2 -1  0  (bias)

→ 가까울수록 덜 페널티
```

---

## 2. 수식

### 2.1 ALiBi Attention

$$\text{Attention}_i = \text{softmax}\left(\frac{q_i K^T}{\sqrt{d}} - m \cdot |i - j|\right) V$$

### 2.2 Head별 기울기

$$m_h = \frac{1}{2^{8/H} \cdot h}$$

```python
# 8 heads 예시
slopes = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]

# 다양한 "시야" 제공
# 작은 slope → 긴 범위
# 큰 slope → 짧은 범위
```

---

## 3. 구현

```python
def alibi_attention(Q, K, V, num_heads):
    # Attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # ALiBi bias
    seq_len = Q.size(-2)
    positions = torch.arange(seq_len)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)
    distance = -torch.abs(distance).float()

    # Head별 slope
    slopes = get_slopes(num_heads)  # [1/2, 1/4, ...]
    bias = slopes.view(-1, 1, 1) * distance

    # Bias 추가
    scores = scores + bias

    return F.softmax(scores, dim=-1) @ V
```

---

## 4. 길이 외삽

### 4.1 실험 결과

```
학습: 1024 토큰
추론: 2048, 4096, ... 토큰

ALiBi perplexity:
- 1024: 18.5 (학습 길이)
- 2048: 18.7 (외삽)
- 4096: 19.1 (외삽)
- 8192: 19.8 (외삽)

Sinusoidal은 2048부터 급격히 악화!
```

### 4.2 왜 외삽이 잘 되는가?

```
선형 bias = 거리에 비례하는 페널티

학습 시 본 적 없는 긴 거리도
같은 규칙 적용 가능
```

---

## 5. 장점과 한계

### 5.1 장점

```
1. 길이 외삽 우수
2. Position embedding 불필요
3. 구현 간단
4. 추가 파라미터 없음
```

### 5.2 한계

```
1. RoPE 대비 일부 태스크에서 약간 낮은 성능
2. Causal attention에 최적화
3. 양방향 attention에서는 다른 설정 필요
```

---

## 6. 사용 모델

| 모델 | 위치 인코딩 |
|------|-------------|
| BLOOM | ALiBi |
| MPT | ALiBi |
| Falcon | ALiBi |
| LLaMA | RoPE |

---

## 7. 핵심 요약

### 기억해야 할 것들

1. **핵심**: 거리 기반 선형 페널티
2. **장점**: 뛰어난 길이 외삽
3. **구현**: Attention score에 bias 추가
4. **사용**: BLOOM, Falcon

### 핵심 수식

$$\text{Attention}_i \propto \exp\left(\frac{q_i k_j^T}{\sqrt{d}} - m|i-j|\right)$$

---

## 참고 자료

1. [ALiBi 논문](https://arxiv.org/abs/2108.12409)

---

*이전 리뷰: [RoPE](./001_RoPE.md)*
*다음 리뷰: [GQA](./003_GQA.md)*
