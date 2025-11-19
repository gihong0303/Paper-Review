# Mamba-2: Transformers are SSMs

**논문 발표**: 2024년
**저자**: Tri Dao, Albert Gu
**소속**: Princeton University, Carnegie Mellon University
**논문 링크**: [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)

---

## 한 줄 요약
> Structured State Space Duality를 통해 SSM과 Attention의 이론적 연결을 밝히고, 2-8배 빠른 Mamba-2 아키텍처 제시

---

## 1. State Space Duality (SSD)

### 1.1 핵심 발견

```
SSM ≈ 특수한 형태의 Attention!

둘 다 다음 형태로 표현 가능:
y = M x

M은 구조화된 행렬
```

### 1.2 수식적 연결

SSM:
$$y_t = \sum_{s=1}^{t} C_t A_{t:s} B_s x_s$$

Attention:
$$y_t = \sum_{s=1}^{t} \text{softmax}(Q_t K_s^T) V_s$$

둘 다 "가중합" 형태!

---

## 2. Mamba-2 개선

### 2.1 State Space Dual (SSD) Layer

```python
# 더 단순하고 효율적인 구조
class SSDLayer(nn.Module):
    def __init__(self, d_model, d_state, nheads):
        self.nheads = nheads
        self.head_dim = d_model // nheads

        # 간소화된 파라미터
        self.A = nn.Parameter(-torch.ones(nheads))

    def forward(self, x, B, C):
        # Matrix multiply formulation
        # 더 효율적인 연산
        ...
```

### 2.2 Multi-Head SSM

```
Mamba-1: 단일 head
Mamba-2: Multi-head

→ 더 풍부한 표현력
→ Tensor Core 활용 가능
```

---

## 3. 속도 향상

### 3.1 벤치마크

| 시퀀스 길이 | Mamba-1 | Mamba-2 |
|-------------|---------|---------|
| 2K | 1x | 2x |
| 8K | 1x | 4x |
| 16K | 1x | 8x |

### 3.2 이유

```
1. 더 단순한 구조
2. Matrix multiply로 표현 가능
3. Tensor Core 활용
4. 더 나은 메모리 패턴
```

---

## 4. 성능

### 4.1 언어 모델링

| 모델 | 파라미터 | Pile PPL |
|------|----------|----------|
| Mamba-1 | 2.8B | 6.2 |
| **Mamba-2** | **2.7B** | **5.8** |
| Transformer++ | 2.7B | 5.9 |

더 빠르면서 더 좋은 성능!

---

## 5. 핵심 요약

### 기억해야 할 것들

1. **이론**: SSM ≈ Structured Attention
2. **속도**: 2-8배 향상
3. **구조**: Multi-head SSM
4. **성능**: Transformer 능가

### SSD 의의

```
SSM과 Attention이 같은 프레임워크!
→ 두 분야의 기술 교류 가능
→ 새로운 아키텍처 설계 영감
```

---

## 참고 자료

1. [Mamba-2 논문](https://arxiv.org/abs/2405.21060)

---

*이전 리뷰: [Jamba](./003_Jamba.md)*
*다음 리뷰: [Hyena](./005_Hyena.md)*
