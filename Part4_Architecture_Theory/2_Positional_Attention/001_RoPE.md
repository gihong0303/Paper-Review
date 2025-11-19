# RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)

**논문 발표**: 2021년
**저자**: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
**소속**: Zhuiyi Technology
**논문 링크**: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

---

## 한 줄 요약
> 절대 위치를 회전 행렬로 인코딩하여 상대 위치 정보를 자연스럽게 얻는 위치 인코딩 방법으로, LLaMA, GPT-NeoX 등 현대 LLM의 표준

---

## 1. 핵심 아이디어

### 1.1 목표

```
원하는 것:
⟨f(q, m), f(k, n)⟩ = g(q, k, m-n)

→ Attention이 상대 위치 m-n만 의존
```

### 1.2 해결책: 회전

복소수 공간에서 회전:
$$f(x, m) = e^{im\theta}x$$

---

## 2. 수학적 유도

### 2.1 2D 회전

$$R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

### 2.2 RoPE 공식

$$RoPE(x, m) = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \cos m\theta_2 \\ \cos m\theta_2 \\ \vdots \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \\ -x_4 \\ x_3 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \sin m\theta_2 \\ \sin m\theta_2 \\ \vdots \end{pmatrix}$$

### 2.3 주파수

$$\theta_i = 10000^{-2i/d}$$

---

## 3. 구현

```python
def rotary_embedding(x, seq_len):
    """
    x: [batch, seq, heads, dim]
    """
    dim = x.shape[-1]
    # 주파수 계산
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    # 위치
    positions = torch.arange(seq_len).float()
    # 각도
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)

    # cos, sin
    cos = angles.cos()
    sin = angles.sin()

    # 회전 적용
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1).flatten(-2)

    return rotated
```

---

## 4. 장점

### 4.1 상대 위치

```
⟨RoPE(q, m), RoPE(k, n)⟩
= ⟨q, k⟩ · cos((m-n)θ) + ...

→ 자연스럽게 상대 위치 의존!
```

### 4.2 길이 외삽

```
학습: 2048 토큰
추론: 4096+ 토큰 가능

→ 외삽 가능 (제한적)
```

### 4.3 효율성

```
- 추가 파라미터 없음
- 연산 오버헤드 최소
- 캐싱 가능
```

---

## 5. 현대 LLM에서의 사용

| 모델 | RoPE 사용 | θ base |
|------|-----------|--------|
| LLaMA 1 | Yes | 10,000 |
| LLaMA 2 | Yes | 10,000 |
| LLaMA 3 | Yes | 500,000 |
| Mistral | Yes | 10,000 |
| Qwen | Yes | 1,000,000 |

---

## 6. 핵심 요약

### 기억해야 할 것들

1. **핵심**: 절대 위치를 회전으로 인코딩
2. **효과**: 상대 위치 정보 자연스럽게 획득
3. **장점**: 효율적, 외삽 가능, 파라미터 없음
4. **사용**: LLaMA, Mistral 등 현대 LLM 표준

### 핵심 수식

$$f(x, m) = x \cdot e^{im\theta}$$

---

## 참고 자료

1. [RoPE 논문](https://arxiv.org/abs/2104.09864)
2. [Eleuther Blog](https://blog.eleuther.ai/rotary-embeddings/)

---

*다음 리뷰: [ALiBi](./002_ALiBi.md)*
