# RWKV: Reinventing RNNs for the Transformer Era

**논문 발표**: 2023년 (EMNLP 2023 Findings)
**저자**: Bo Peng, Eric Alcaide, Quentin Anthony, et al.
**소속**: RWKV Foundation
**논문 링크**: [arXiv:2305.13048](https://arxiv.org/abs/2305.13048)
**공식 구현**: [GitHub](https://github.com/BlinkDL/RWKV-LM)

---

## 한 줄 요약
> Transformer의 병렬 학습과 RNN의 효율적 추론을 결합하여, O(N) 복잡도로 무한 컨텍스트를 처리하는 새로운 아키텍처

---

## 1. Transformer vs RNN vs RWKV

### 1.1 비교

| 특성 | Transformer | RNN | RWKV |
|------|-------------|-----|------|
| 학습 | 병렬 | 순차 | **병렬** |
| 추론 | O(N²) | O(1) | **O(1)** |
| 컨텍스트 | 제한 | 이론적 무한 | **무한** |

### 1.2 핵심 아이디어

```
RWKV = Receptance Weighted Key Value

RNN처럼 동작하지만
Transformer처럼 학습 가능
```

---

## 2. 아키텍처

### 2.1 Time Mixing

$$wkv_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w+k_i} v_i + e^{u+k_t} v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w+k_i} + e^{u+k_t}}$$

### 2.2 Channel Mixing

$$r_t = \sigma(W_r \cdot (\mu_r x_t + (1-\mu_r) x_{t-1}))$$
$$o_t = r_t \odot (W_v \cdot \max(k_t, 0)^2)$$

---

## 3. 구현

```python
def rwkv_time_mixing(x, state):
    """
    RNN 모드 추론 (O(1) per token)
    """
    # Previous state
    prev_x, aa, bb, pp = state

    # Current token
    k = W_k @ x
    v = W_v @ x
    r = sigmoid(W_r @ x)

    # Weighted sum
    ww = u + k
    p = max(pp, ww)
    e1 = exp(pp - p)
    e2 = exp(ww - p)

    output = r * (e1 * aa + e2 * v) / (e1 * bb + e2)

    # Update state
    ww = w + pp
    p = max(ww, k)
    e1 = exp(ww - p)
    e2 = exp(k - p)
    new_aa = e1 * aa + e2 * v
    new_bb = e1 * bb + e2

    return output, (x, new_aa, new_bb, p)
```

---

## 4. 장점

### 4.1 학습 효율

```
병렬화 가능:
- 전체 시퀀스를 한 번에
- Transformer와 같은 속도
```

### 4.2 추론 효율

```
Constant memory:
- 상태만 유지
- O(1) per token
- 무한 컨텍스트
```

---

## 5. 실험 결과

| 모델 | 파라미터 | LAMBADA |
|------|----------|---------|
| GPT-Neo | 2.7B | 62.2 |
| GPT-J | 6B | 69.7 |
| **RWKV** | **7.4B** | **67.2** |

Transformer와 유사한 성능!

---

## 6. 핵심 요약

### 기억해야 할 것들

1. **핵심**: RNN + Transformer 장점 결합
2. **효율**: O(1) 추론, 병렬 학습
3. **컨텍스트**: 무한 길이 가능
4. **성능**: Transformer에 근접

---

## 참고 자료

1. [RWKV 논문](https://arxiv.org/abs/2305.13048)
2. [GitHub](https://github.com/BlinkDL/RWKV-LM)

---

*다음 리뷰: [Mamba](./002_Mamba.md)*
