# BitNet: Scaling 1-bit Transformers for Large Language Models

**논문 발표**: 2023년
**저자**: Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huishuai Zhang, Dongdong Zhang, Furu Wei
**소속**: Microsoft Research
**논문 링크**: [arXiv:2310.11453](https://arxiv.org/abs/2310.11453)

---

## 한 줄 요약
> 1비트 가중치({-1, 1})와 8비트 활성화를 사용하는 BitLinear 레이어로 Transformer를 구성하여, FP16 대비 메모리와 에너지를 크게 절약하면서 경쟁력 있는 성능 달성

---

## 1. 핵심 아이디어: 1-bit Weights

### 1.1 극단적 양자화

```
FP16: 65,536 가지 값
INT8: 256 가지 값
INT4: 16 가지 값
INT1: 2 가지 값 {-1, +1}  ← BitNet
```

### 1.2 Binary Quantization

가중치를 이진화:
$$\tilde{W} = \text{Sign}(W - \alpha)$$

$$\alpha = \frac{1}{nm}\sum_{ij}W_{ij}$$

결과: 모든 가중치가 {-1, +1}

### 1.3 BitLinear 레이어

```python
def BitLinear(x, W):
    # 가중치 이진화
    W_binary = sign(W - W.mean())  # {-1, +1}
    beta = W.abs().mean()          # 스케일 팩터

    # 활성화 양자화 (8bit)
    x_quant = quantize_8bit(x)
    gamma = x.abs().max()

    # 행렬 곱 (정수 연산)
    y = x_quant @ W_binary

    # 역양자화
    return y * (beta * gamma / 127)
```

---

## 2. 학습 방법

### 2.1 Straight-Through Estimator (STE)

Sign 함수는 미분 불가 → STE로 근사:

```python
# Forward: 이진화
W_binary = sign(W)

# Backward: Gradient를 그대로 전달
W.grad = W_binary.grad  # STE
```

### 2.2 학습 안정화

- **Group Normalization**: LayerNorm 전에 적용
- **Gradient Scaling**: 큰 초기 gradient 조절

---

## 3. 실험 결과

### 3.1 Perplexity

| 모델 크기 | FP16 Transformer | BitNet |
|-----------|------------------|--------|
| 125M | 28.1 | 30.2 |
| 350M | 21.9 | 22.8 |
| 1.3B | 14.1 | 15.4 |

### 3.2 효율성

| 메트릭 | FP16 | BitNet | 절약 |
|--------|------|--------|------|
| 메모리 | 기준 | ~11× | 91% |
| 에너지 | 기준 | ~7× | 86% |

---

## 4. 한계점

1. **정확도 격차**: FP16 대비 여전히 차이 존재
2. **학습 필요**: PTQ 불가, 처음부터 학습 필요
3. **하드웨어 지원**: 전용 하드웨어 필요

---

## 5. 핵심 요약

- **핵심**: 가중치를 {-1, +1}로 극단 양자화
- **장점**: 메모리/에너지 대폭 절약
- **단점**: 성능 저하, 학습 필요
- **의의**: 극단적 양자화의 가능성 제시

→ 후속 연구 BitNet b1.58이 이 문제들을 해결

---

## 참고 자료

1. [BitNet 논문](https://arxiv.org/abs/2310.11453)
2. [Microsoft Research](https://www.microsoft.com/en-us/research/)

---

*이전 리뷰: [SqueezeLLM](./005_SqueezeLLM.md)*
*다음 리뷰: [BitNet b1.58](./007_BitNet_b158.md)*
