# Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads

**논문 발표**: 2024년
**저자**: Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao
**소속**: Princeton, UIUC, Together AI
**논문 링크**: [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
**공식 구현**: [GitHub](https://github.com/FasterDecoding/Medusa)

---

## 한 줄 요약
> 별도의 Draft 모델 없이, 기존 LLM에 추가 Decoding Head들을 부착하여 여러 후보 토큰을 동시에 예측하고 Tree Attention으로 검증하여 2-3배 속도 향상 달성

---

## 1. 핵심 아이디어

### 1.1 기존 Speculative Decoding의 한계

```
문제:
- 별도의 Draft 모델 필요
- 두 모델을 메모리에 동시 로드
- Draft 모델 선택 어려움
```

### 1.2 Medusa의 해결책

> "Draft 모델 대신, 여러 개의 Decoding Head를 추가하자!"

```
기존 LLM:
[LLM] → [LM Head] → 다음 토큰 1개

Medusa:
[LLM] → [LM Head]     → 토큰 t+1
      → [Medusa Head 1] → 토큰 t+2
      → [Medusa Head 2] → 토큰 t+3
      → [Medusa Head 3] → 토큰 t+4
```

### 1.3 장점

- **메모리 효율**: Head만 추가 (원본 대비 <1% 파라미터)
- **호환성**: 기존 모델에 바로 적용
- **단순함**: Draft 모델 선택 불필요

---

## 2. 아키텍처

### 2.1 Medusa Head 구조

```python
class MedusaHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):
        x = self.linear(hidden_states)
        x = self.ln(x)
        x = F.silu(x)
        logits = self.lm_head(x)
        return logits
```

### 2.2 전체 구조

```
Input → [Transformer Layers] → Hidden States
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        [Original LM Head]   [Medusa Head 1]       [Medusa Head k]
              │                     │                     │
              ▼                     ▼                     ▼
         Logits t+1            Logits t+2            Logits t+k+1
```

---

## 3. Tree Attention

### 3.1 후보 생성

각 Head의 top-k 토큰 조합:

```
Head 1 top-2: [A, B]
Head 2 top-2: [C, D]
Head 3 top-2: [E, F]

후보 트리:
    root
    ├── A
    │   ├── C
    │   │   ├── E
    │   │   └── F
    │   └── D
    │       ├── E
    │       └── F
    └── B
        ├── C
        │   ├── E
        │   └── F
        └── D
            ├── E
            └── F
```

### 3.2 Tree Attention 검증

Attention mask를 Tree 구조에 맞게 설정:

```python
def create_tree_attention_mask(tree_structure):
    """
    Tree 구조에 따른 causal mask 생성
    각 노드는 자신의 조상만 attend
    """
    n_candidates = len(tree_structure)
    mask = torch.zeros(n_candidates, n_candidates)

    for i, node in enumerate(tree_structure):
        # 자기 자신과 조상에만 attend
        for ancestor in get_ancestors(node):
            mask[i, ancestor] = 1

    return mask
```

### 3.3 검증 및 수용

```python
def verify_candidates(model, candidates, tree_mask):
    # Tree attention으로 병렬 검증
    logits = model.forward(candidates, attention_mask=tree_mask)

    # 가장 긴 일치 경로 찾기
    for path in tree_paths:
        if all(path[i] == argmax(logits[i]) for i in path):
            accepted_path = path
            break

    return accepted_path
```

---

## 4. 학습 방법

### 4.1 Medusa-1 (Fine-tuning)

Original LLM을 freeze하고 Head만 학습:

```python
# Frozen base model
for param in base_model.parameters():
    param.requires_grad = False

# Trainable heads
for head in medusa_heads:
    for param in head.parameters():
        param.requires_grad = True

# 학습 데이터: ShareGPT 등
```

### 4.2 Medusa-2 (Joint training)

Base model과 Head를 함께 학습:
- 더 좋은 성능
- 더 많은 연산 필요

---

## 5. 실험 결과

### 5.1 속도 향상

| 모델 | Medusa-1 | Medusa-2 |
|------|----------|----------|
| Vicuna-7B | 2.2× | **2.3×** |
| Vicuna-13B | 2.3× | **2.4×** |
| Vicuna-33B | 1.9× | **2.1×** |

### 5.2 품질 유지

| 모델 | 원본 | Medusa |
|------|------|--------|
| Vicuna-7B (MT-Bench) | 6.2 | 6.2 |
| Vicuna-13B (MT-Bench) | 6.7 | 6.6 |

### 5.3 메모리 오버헤드

```
Base Model: 100%
+ Medusa Heads: +0.5-1%

vs Speculative Decoding:
Base Model: 100%
+ Draft Model: +10-50%
```

---

## 6. 쉬운 예시

### 6.1 시험 예측 비유

**기존**: 한 문제씩 풀기

**Speculative Decoding**:
- 연습생이 먼저 5문제 풀기
- 선생님이 검토

**Medusa**:
- 선생님이 1번 문제를 보면서
- 동시에 2, 3, 4번 답도 예측
- Tree 형태로 후보들 검증

---

## 7. 구현 및 사용

```python
from medusa import MedusaModel

# 모델 로드
model = MedusaModel.from_pretrained(
    "FasterDecoding/medusa-vicuna-7b-v1.3",
    medusa_num_heads=4,
    medusa_num_layers=1
)

# 추론
output = model.generate(
    input_ids,
    max_length=256,
    temperature=0.7
)
```

---

## 8. 핵심 요약

### 기억해야 할 것들

1. **핵심**: Draft 모델 대신 여러 Head 추가
2. **장점**: 메모리 효율적, 간단한 적용
3. **방법**: Tree Attention으로 병렬 검증
4. **결과**: 2-2.5× 속도 향상

### vs Speculative Decoding

| 측면 | Speculative | Medusa |
|------|-------------|--------|
| 추가 모델 | 필요 (Draft) | 불필요 |
| 메모리 | +10-50% | **+<1%** |
| 적용 난이도 | 중간 | **쉬움** |
| 속도 | 2-3× | 2-2.5× |

---

## 참고 자료

1. [Medusa 논문](https://arxiv.org/abs/2401.10774)
2. [GitHub](https://github.com/FasterDecoding/Medusa)
3. [Together AI Blog](https://www.together.ai/)

---

*이전 리뷰: [Speculative Sampling](./002_Speculative_Sampling.md)*
*다음 리뷰: [Eagle](./004_Eagle.md)*
