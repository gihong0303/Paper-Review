# DoRA: Weight-Decomposed Low-Rank Adaptation

**논문 발표**: 2024년
**저자**: Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen
**소속**: NVIDIA, National Taiwan University, HKUST
**논문 링크**: [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)

---

## 한 줄 요약
> 가중치를 Magnitude와 Direction으로 분해하여 학습함으로써, LoRA의 학습 용량 문제를 해결하고 Full Fine-tuning에 더 가까운 성능 달성

---

## 1. LoRA의 한계

### 1.1 관찰

LoRA가 Full FT보다 성능이 낮은 경우가 있음

**원인 분석**: LoRA는 magnitude와 direction을 동시에 학습하려 함
→ 학습 용량이 부족

### 1.2 Weight Decomposition

가중치를 크기와 방향으로 분해:
$$W = m \cdot \frac{V}{\|V\|_c}$$

- $m$: magnitude (크기)
- $V / \|V\|_c$: direction (방향)

---

## 2. DoRA 방법

### 2.1 수식

$$W' = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$$

- $m$: 학습 가능한 magnitude 벡터
- $BA$: LoRA와 동일
- Direction만 LoRA로 학습

### 2.2 구현

```python
class DoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank):
        # LoRA 파라미터
        self.A = nn.Parameter(torch.randn(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

        # Magnitude 파라미터
        self.m = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        # Direction
        W_prime = self.W0 + self.B @ self.A
        direction = W_prime / W_prime.norm(dim=0, keepdim=True)

        # Output
        return x @ (self.m * direction)
```

---

## 3. 실험 결과

### 3.1 vs LoRA

| 태스크 | LoRA | DoRA | 개선 |
|--------|------|------|------|
| Commonsense | 80.4 | **81.9** | +1.5 |
| Visual Inst | 81.4 | **82.6** | +1.2 |
| Image/Video | 76.8 | **78.4** | +1.6 |

### 3.2 학습 안정성

DoRA가 더 안정적인 학습 곡선

---

## 4. 핵심 요약

1. **핵심**: Magnitude와 Direction 분리
2. **장점**: LoRA 대비 성능 향상, 학습 안정성
3. **비용**: 약간의 추가 파라미터 (magnitude vector)

---

## 참고 자료

1. [DoRA 논문](https://arxiv.org/abs/2402.09353)

---

*이전 리뷰: [QLoRA](./002_QLoRA.md)*
*다음 리뷰: [Prefix-Tuning](./004_Prefix-Tuning.md)*
