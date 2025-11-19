# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**논문 발표**: 2023년
**저자**: Albert Gu, Tri Dao
**소속**: Carnegie Mellon University, Princeton University
**논문 링크**: [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
**공식 구현**: [GitHub](https://github.com/state-spaces/mamba)

---

## 한 줄 요약
> Selective State Space Model로 O(N²) Transformer를 O(N) 복잡도로 대체하면서, 언어 모델링에서 Transformer에 필적하는 성능 달성

---

## 1. State Space Model (SSM)

### 1.1 기본 수식

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

이산화:
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t$$

### 1.2 문제: Content-Aware 불가

```
기존 SSM:
A, B, C가 고정 → 입력에 무관

해결:
Selective SSM
→ A, B, C를 입력에 따라 동적 결정
```

---

## 2. Selective State Space

### 2.1 핵심 아이디어

$$B_t = s_B(x_t), \quad C_t = s_C(x_t), \quad \Delta_t = s_\Delta(x_t)$$

```python
def selective_ssm(x, A, D):
    # 입력에 따라 파라미터 동적 생성
    B = linear_B(x)  # [B, L, N]
    C = linear_C(x)  # [B, L, N]
    delta = softplus(linear_delta(x))  # [B, L, D]

    # Discretization
    A_bar = exp(delta * A)  # [B, L, D, N]
    B_bar = delta * B

    # Selective scan
    y = selective_scan(x, A_bar, B_bar, C, D)
    return y
```

### 2.2 Selection = Attention?

```
Selection의 효과:
- 관련 정보 선택
- 무관한 정보 무시

Attention과 유사하지만 O(N)!
```

---

## 3. Mamba Block

```python
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, expand):
        self.in_proj = nn.Linear(d_model, 2 * expand * d_model)
        self.conv1d = nn.Conv1d(expand * d_model, expand * d_model, 4)
        self.ssm = SelectiveSSM(expand * d_model, d_state)
        self.out_proj = nn.Linear(expand * d_model, d_model)

    def forward(self, x):
        # Dual path
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Conv + SSM
        x = self.conv1d(x)
        x = self.ssm(silu(x))

        # Gate and project
        y = x * silu(z)
        return self.out_proj(y)
```

---

## 4. Hardware-Aware 구현

### 4.1 문제

```
Selective SSM:
- 파라미터가 시퀀스마다 다름
- 일반적인 convolution 불가
- 메모리 bandwidth 병목
```

### 4.2 해결: Kernel Fusion

```
GPU 최적화:
1. 중간 상태를 SRAM에 유지
2. Fused kernel로 메모리 접근 최소화
3. Recomputation으로 backward 최적화
```

---

## 5. 실험 결과

### 5.1 언어 모델링

| 모델 | 파라미터 | PPL (Pile) |
|------|----------|------------|
| Transformer++ | 1.4B | 8.5 |
| RWKV | 1.5B | 8.8 |
| **Mamba** | **1.4B** | **8.3** |

### 5.2 Long Context

```
Mamba 강점:
- 긴 시퀀스에서 성능 유지
- 메모리 효율적
- 속도 빠름
```

---

## 6. 핵심 요약

### 기억해야 할 것들

1. **핵심**: Selective State Space Model
2. **복잡도**: O(N) (vs Transformer O(N²))
3. **Selection**: 입력에 따라 파라미터 동적 결정
4. **성능**: Transformer에 필적

### 수식 요약

$$y = \text{SSM}(A, B_t, C_t, D)(x)$$

여기서 $B_t, C_t$가 입력 의존!

---

## 참고 자료

1. [Mamba 논문](https://arxiv.org/abs/2312.00752)
2. [GitHub](https://github.com/state-spaces/mamba)

---

*이전 리뷰: [RWKV](./001_RWKV.md)*
*다음 리뷰: [Jamba](./003_Jamba.md)*
