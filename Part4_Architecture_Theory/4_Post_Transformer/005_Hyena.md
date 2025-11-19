# Hyena Hierarchy: Towards Larger Convolutional Language Models

**논문 발표**: 2023년 (ICML 2023)
**저자**: Michael Poli, Stefano Massaroli, Eric Nguyen, et al.
**소속**: Stanford University, Together AI
**논문 링크**: [arXiv:2302.10866](https://arxiv.org/abs/2302.10866)

---

## 한 줄 요약
> Attention 없이 긴 convolution과 gating만으로 O(N log N) 복잡도를 달성하면서 Transformer에 근접한 성능

---

## 1. Attention 없이 가능한가?

### 1.1 Attention의 문제

```
O(N²) 복잡도
→ 긴 시퀀스 불가능
→ 메모리 병목
```

### 1.2 Hyena의 해결

```
Long Convolution + Gating:
- O(N log N) 복잡도
- 긴 시퀀스 처리 가능
- Data-dependent (like attention)
```

---

## 2. Hyena Operator

### 2.1 수식

$$y = (h * (v \odot (h * (v \odot x))))$$

```
h: implicit long convolution filter
v: input-dependent gating
*: convolution
⊙: element-wise multiplication
```

### 2.2 구조

```python
def hyena_operator(x, order=2):
    """
    order: 반복 횟수 (보통 2)
    """
    # Projections
    v = proj_v(x)
    q = proj_q(x)

    # Implicit convolution filter
    h = generate_filter(seq_len)

    # Hyena recurrence
    y = v
    for _ in range(order):
        y = fft_conv(h, y * v)

    return y * q
```

---

## 3. Long Convolution

### 3.1 Implicit Filter

```
Filter를 직접 저장하지 않고
작은 네트워크로 생성:

h[t] = MLP(positional_encoding(t))

→ 메모리 효율적
→ 무한 길이 가능
```

### 3.2 FFT Convolution

$$h * x = \text{IFFT}(\text{FFT}(h) \odot \text{FFT}(x))$$

복잡도: O(N log N)

---

## 4. Data-Dependent

### 4.1 vs 일반 Convolution

```
일반 Conv: 같은 filter 적용
Hyena: gating으로 input-dependent

v ⊙ x 에서:
v가 입력에 따라 변함
→ 다른 위치에 다른 가중치
```

---

## 5. 실험 결과

### 5.1 언어 모델링

| 모델 | 파라미터 | WikiText PPL |
|------|----------|--------------|
| Transformer | 125M | 28.5 |
| **Hyena** | **125M** | **29.1** |

근접한 성능!

### 5.2 Long Range Arena

```
긴 시퀀스 태스크에서:
Hyena > Transformer

긴 컨텍스트 강점!
```

---

## 6. 핵심 요약

### 기억해야 할 것들

1. **핵심**: Attention 없이 convolution만
2. **복잡도**: O(N log N)
3. **Data-dependent**: Gating으로 구현
4. **성능**: Transformer에 근접

### Hyena 특징

```
장점:
- 긴 시퀀스 효율적
- 병렬 학습 가능
- 메모리 효율적

한계:
- 짧은 시퀀스에서 Attention이 더 좋음
- 복잡한 추론 태스크에서 약간 부족
```

---

## 참고 자료

1. [Hyena 논문](https://arxiv.org/abs/2302.10866)

---

*이전 리뷰: [Mamba-2](./004_Mamba-2.md)*
*다음 섹션: [Reasoning & Chain of Thought](../5_Reasoning_CoT/)*
