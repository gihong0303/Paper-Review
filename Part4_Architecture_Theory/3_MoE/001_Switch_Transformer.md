# Switch Transformers: Scaling to Trillion Parameter Models

**논문 발표**: 2021년 (JMLR 2022)
**저자**: William Fedus, Barret Zoph, Noam Shazeer
**소속**: Google Brain
**논문 링크**: [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)

---

## 한 줄 요약
> 각 토큰이 하나의 expert만 선택하는 Sparse MoE로 1조 파라미터 모델을 효율적으로 학습, MoE의 실용화를 이끈 기념비적 연구

---

## 1. Mixture of Experts (MoE)

### 1.1 Dense vs Sparse

```
Dense Model:
- 모든 토큰이 모든 파라미터 사용
- 파라미터 ↑ → 연산 ↑

Sparse MoE:
- 각 토큰이 일부 expert만 사용
- 파라미터 ↑ → 연산 거의 동일!
```

### 1.2 구조

```
입력 → Router → Expert 선택 → 연산 → 출력

Router: 어떤 expert를 선택할지 결정
Expert: FFN 레이어 (여러 개 존재)
```

---

## 2. Switch Transformer

### 2.1 핵심: Top-1 Routing

```
기존 MoE: Top-2 experts
Switch: Top-1 expert만!

장점:
- 더 간단한 라우팅
- 통신 비용 절감
- 배치 효율성
```

### 2.2 수식

$$y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)$$

Switch에서:
$$g_i(x) = \begin{cases} 1 & \text{if } i = \arg\max(\text{Router}(x)) \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. 구현

```python
class SwitchFFN(nn.Module):
    def __init__(self, d_model, d_ff, num_experts):
        super().__init__()
        self.num_experts = num_experts

        # Router
        self.router = nn.Linear(d_model, num_experts)

        # Experts (각각 FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # Router probabilities
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=-1)

        # Top-1 selection
        expert_idx = routing_weights.argmax(dim=-1)

        # Route to selected expert
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (expert_idx == i)
            if mask.any():
                output[mask] = self.experts[i](x[mask])

        return output
```

---

## 4. Load Balancing

### 4.1 문제

```
모든 토큰이 같은 expert로 → 불균형

Expert 1: 90% 토큰 → 과부하
Expert 2-N: 10% 토큰 → 유휴
```

### 4.2 Auxiliary Loss

$$\mathcal{L}_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

- $f_i$: expert $i$로 라우팅된 토큰 비율
- $P_i$: expert $i$의 평균 routing 확률

---

## 5. 스케일링 결과

### 5.1 학습 속도

```
T5-Base vs Switch-Base (같은 FLOPs):
- Switch가 7배 빠르게 수렴

더 많은 파라미터 → 더 빠른 학습
```

### 5.2 성능

| 모델 | 파라미터 | C4 PPL |
|------|----------|--------|
| T5-Base | 220M | 4.9 |
| Switch-Base | 3.8B | 4.5 |
| T5-Large | 739M | 4.3 |
| Switch-Large | 26B | 3.9 |

---

## 6. 분산 학습

### 6.1 Expert Parallelism

```
Expert를 다른 GPU에 분산:

GPU 0: Expert 0, 1
GPU 1: Expert 2, 3
GPU 2: Expert 4, 5
GPU 3: Expert 6, 7

토큰 → 해당 GPU로 전송 → 연산 → 반환
```

---

## 7. 핵심 요약

### 기억해야 할 것들

1. **핵심**: 각 토큰이 1개 expert만 사용
2. **효과**: 파라미터 ↑, 연산 동일
3. **학습**: 7배 빠른 수렴
4. **결과**: 1조 파라미터 모델

### Switch 설정

| 항목 | 값 |
|------|-----|
| Experts | 128 |
| Top-k | 1 |
| Capacity | 1.25× |

---

## 참고 자료

1. [Switch Transformer 논문](https://arxiv.org/abs/2101.03961)

---

*다음 리뷰: [Mixtral](./002_Mixtral.md)*
