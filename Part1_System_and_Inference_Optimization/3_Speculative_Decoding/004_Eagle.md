# Eagle: Speculative Sampling Requires Rethinking Feature Uncertainty

**논문 발표**: 2024년 (ICML 2024)
**저자**: Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang
**소속**: Peking University, Microsoft Research
**논문 링크**: [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
**공식 구현**: [GitHub](https://github.com/SafeAILab/EAGLE)

---

## 한 줄 요약
> 이전 Hidden state와 토큰 임베딩을 함께 사용하는 Autoregressive Draft Head로, 기존 Speculative Decoding 방법들 대비 SOTA 성능(3× 속도 향상, 분포 유지)을 달성

---

## 1. 기존 방법들의 한계

### 1.1 Speculative Decoding

```
문제: 별도 Draft 모델 필요
- 메모리 오버헤드
- 분포 불일치
```

### 1.2 Medusa

```
문제: Feature uncertainty 무시
- 각 Head가 독립적으로 예측
- 이전 예측의 불확실성 미반영
```

---

## 2. Eagle의 핵심 아이디어

### 2.1 Feature Uncertainty

> "다음 토큰 예측 시, 이전 토큰의 **불확실성**을 고려해야 한다"

```
Medusa:  Hidden[t] → Head → Token[t+1]
         Hidden[t] → Head → Token[t+2] (독립)

Eagle:   Hidden[t] + Token[t+1] → AutoReg → Token[t+2]
         Hidden[t] + Token[t+1] + Token[t+2] → AutoReg → Token[t+3]
```

### 2.2 Autoregressive Draft Head

```python
class EagleDraftHead(nn.Module):
    def forward(self, hidden_state, prev_token_embed):
        # 이전 hidden과 토큰 임베딩 결합
        combined = self.fc(torch.cat([hidden_state, prev_token_embed]))

        # 경량 Transformer 레이어
        for layer in self.transformer_layers:
            combined = layer(combined)

        # LM Head로 예측
        logits = self.lm_head(combined)
        return logits
```

---

## 3. 아키텍처

### 3.1 전체 구조

```
Base LLM: [Transformer Layers] → Hidden[t]
                                    │
                                    ▼
            ┌──────────────────────────────────────┐
            │         Eagle Draft Head             │
            │                                      │
            │  Hidden[t] + Embed[t] → FC           │
            │       │                              │
            │       ▼                              │
            │  [Transformer Layer] → Logits[t+1]   │
            │       │                              │
            │       ▼                              │
            │  + Embed[t+1] → [Trans] → Logits[t+2]│
            │       │                              │
            │       ▼                              │
            │  + Embed[t+2] → [Trans] → Logits[t+3]│
            │                                      │
            └──────────────────────────────────────┘
```

### 3.2 특징

1. **Autoregressive**: 이전 예측을 다음 예측에 반영
2. **경량화**: 1-2개의 Transformer 레이어만 사용
3. **Feature 재사용**: Base model의 hidden state 활용

---

## 4. Tree-based Decoding

### 4.1 Draft Tree 생성

```python
def generate_draft_tree(eagle_head, hidden, depth, width):
    candidates = []

    # 첫 번째 레벨
    logits = eagle_head(hidden, embed(current_token))
    top_k = logits.topk(width)

    for token in top_k:
        # 재귀적으로 다음 레벨 생성
        sub_tree = generate_draft_tree(
            eagle_head,
            eagle_head.next_hidden,
            depth - 1,
            width
        )
        candidates.append((token, sub_tree))

    return candidates
```

### 4.2 검증

Base model로 Tree의 모든 경로를 병렬 검증

---

## 5. 학습

### 5.1 학습 목표

다음 토큰 예측 (standard LM objective):

```python
loss = CrossEntropy(
    eagle_head(hidden[:-1], embeds[:-1]),
    target_tokens[1:]
)
```

### 5.2 학습 데이터

ShareGPT 대화 데이터 (68K 샘플)

---

## 6. 실험 결과

### 6.1 속도 향상 비교

| 모델 | Speculative | Medusa | **Eagle** |
|------|-------------|--------|-----------|
| Vicuna-7B | 2.0× | 2.2× | **3.0×** |
| Vicuna-13B | 2.1× | 2.3× | **3.1×** |
| LLaMA2-Chat-7B | 1.9× | 2.1× | **2.8×** |
| LLaMA2-Chat-13B | 2.0× | 2.2× | **3.0×** |
| LLaMA2-Chat-70B | 1.8× | 2.0× | **2.7×** |

### 6.2 수용률

| 방법 | 평균 수용 토큰 |
|------|----------------|
| Speculative | 2.8 |
| Medusa | 2.6 |
| **Eagle** | **3.6** |

### 6.3 품질 (MT-Bench)

| 모델 | 원본 | Eagle |
|------|------|-------|
| Vicuna-7B | 6.17 | **6.17** |
| Vicuna-13B | 6.57 | **6.57** |

완벽한 분포 유지!

---

## 7. Eagle-2

### 7.1 개선점

Eagle-2 (2024년 후속):
- **Context-aware draft**: 입력 길이에 따른 적응
- **Dynamic tree**: 상황에 따라 tree 구조 조절
- **3.5-4× 속도 향상**

---

## 8. 쉬운 예시

### 8.1 소설 쓰기 비유

**Medusa**: 첫 문장을 보고 2, 3, 4번째 문장 독립적으로 예측
- 2번째 문장 예측이 틀리면 3, 4번째도 의미 없음

**Eagle**:
- 1번째 문장 → 2번째 문장 예측
- 1번째 + 예측한 2번째 → 3번째 문장 예측
- 연쇄적으로 더 정확한 예측

---

## 9. 구현 및 사용

```python
from eagle import EagleLLM

# 모델 로드
model = EagleLLM.from_pretrained(
    "yuhuili/EAGLE-Vicuna-7B-v1.3"
)

# 추론
output = model.generate(
    input_ids,
    max_new_tokens=256,
    temperature=0.7
)
```

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **핵심 통찰**: Feature uncertainty 고려
2. **방법**: Autoregressive Draft Head
3. **장점**: 높은 수용률, 정확한 분포
4. **결과**: SOTA 3× 속도 향상

### 비교 표

| 방법 | 메모리 | 속도 | 분포 유지 |
|------|--------|------|-----------|
| Speculative | 높음 | 2× | 정확 |
| Medusa | 낮음 | 2.3× | 정확 |
| **Eagle** | **낮음** | **3×** | **정확** |

### 현재 SOTA

Eagle (특히 Eagle-2)는 현재 Speculative Decoding 계열에서 **가장 빠른** 방법

---

## 참고 자료

1. [Eagle 논문](https://arxiv.org/abs/2401.15077)
2. [Eagle-2 논문](https://arxiv.org/abs/2406.16858)
3. [GitHub](https://github.com/SafeAILab/EAGLE)

---

*이전 리뷰: [Medusa](./003_Medusa.md)*
*Part 1 완료! 다음: [Part 2 - Advanced Fine-tuning](../../Part2_Advanced_Finetuning/)*
