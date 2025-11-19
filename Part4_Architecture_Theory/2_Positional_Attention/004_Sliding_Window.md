# Sliding Window Attention (Mistral 7B)

**논문 발표**: 2023년
**저자**: Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, et al.
**소속**: Mistral AI
**논문 링크**: [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
**공식 모델**: [HuggingFace](https://huggingface.co/mistralai)

---

## 한 줄 요약
> 각 토큰이 이전 W개 토큰만 참조하는 윈도우 어텐션으로, O(n²) → O(n·W) 복잡도 감소와 효과적인 긴 컨텍스트 처리

---

## 1. 핵심 아이디어

### 1.1 기존 Full Attention

```
모든 토큰이 모든 이전 토큰 참조:
복잡도: O(n²)
KV 캐시: 전체 시퀀스
```

### 1.2 Sliding Window

```
각 토큰이 이전 W개만 참조:
복잡도: O(n·W)
KV 캐시: W만큼만

예: W = 4096
토큰 5000 → 4997, 4998, 4999, 5000 참조
```

---

## 2. 정보 전파

### 2.1 레이어를 통한 확장

```
Layer 1: 토큰 5000 → 이전 W개 참조
Layer 2: 토큰 5000 → 이전 2W개 정보 접근
Layer 3: 토큰 5000 → 이전 3W개 정보 접근
...

총 32 layers, W=4096:
→ 이론적으로 131K 토큰 정보 접근!
```

### 2.2 수식

Effective context = $L \times W$

```
L = 32 layers
W = 4096 window
→ 128K effective context
```

---

## 3. 구현

```python
def sliding_window_attention(Q, K, V, window_size):
    seq_len = Q.size(-2)

    # Attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Window mask
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        # i번째 토큰은 max(0, i-window_size)부터 i까지만
        mask[i, :max(0, i - window_size + 1)] = False
        mask[i, i+1:] = False  # Causal

    scores = scores.masked_fill(~mask, -1e9)
    weights = F.softmax(scores, dim=-1)

    return weights @ V
```

---

## 4. Rolling Buffer KV Cache

### 4.1 고정 크기 캐시

```python
class RollingKVCache:
    def __init__(self, window_size):
        self.window_size = window_size
        self.cache = None

    def update(self, new_kv, position):
        # 위치 = position % window_size
        idx = position % self.window_size
        self.cache[idx] = new_kv

# KV 캐시가 window_size로 고정!
# 메모리 사용량 일정
```

---

## 5. Mistral 7B 성능

### 5.1 vs LLaMA 2

| 벤치마크 | Mistral 7B | LLaMA 2 13B |
|----------|------------|-------------|
| MMLU | 60.1 | 55.0 |
| HellaSwag | 81.3 | 80.7 |
| WinoGrande | 75.3 | 72.8 |

**7B가 13B를 능가!**

### 5.2 긴 컨텍스트

```
8K context에서도 perplexity 안정
→ Sliding window + 레이어 전파 효과
```

---

## 6. 장점과 한계

### 6.1 장점

```
1. 메모리 효율: O(W) KV 캐시
2. 속도: O(n·W) 복잡도
3. 긴 컨텍스트: 레이어 전파
4. 구현 용이
```

### 6.2 한계

```
1. 직접적 장거리 attention 없음
2. 정보 전파에 레이어 필요
3. 일부 태스크에서 full attention 대비 약간 성능 저하
```

---

## 7. 핵심 요약

### 기억해야 할 것들

1. **핵심**: 이전 W개 토큰만 참조
2. **효과**: O(n·W) 복잡도
3. **KV 캐시**: 고정 크기
4. **전파**: 레이어 통해 긴 범위 정보

### Mistral 설정

| 항목 | 값 |
|------|-----|
| Window | 4096 |
| Layers | 32 |
| Effective | 128K |

---

## 참고 자료

1. [Mistral 7B 논문](https://arxiv.org/abs/2310.06825)

---

*이전 리뷰: [GQA](./003_GQA.md)*
*다음 리뷰: [Ring Attention](./005_Ring_Attention.md)*
