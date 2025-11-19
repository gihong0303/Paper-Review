# Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking

**논문 발표**: 2024년
**저자**: Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, Noah D. Goodman
**소속**: Stanford University
**논문 링크**: [arXiv:2403.09629](https://arxiv.org/abs/2403.09629)

---

## 한 줄 요약
> 모델이 각 토큰 생성 전에 내부적으로 "생각"을 생성하도록 학습하여, 명시적 프롬프팅 없이도 추론 능력 향상

---

## 1. 동기

### 1.1 CoT의 한계

```
CoT 프롬프팅:
- 매번 "Let's think step by step" 필요
- 사용자가 명시적으로 요청해야 함
- 모든 토큰에 적용 불가
```

### 1.2 Quiet-STaR 목표

```
모델 자체가 "생각하는 법"을 학습:
- 프롬프팅 불필요
- 모든 토큰에서 자동 적용
- 내부적 추론 (사용자에게 안 보임)
```

---

## 2. 방법

### 2.1 구조

```
입력: "The answer is"

일반 모델:
"The answer is 42"

Quiet-STaR:
"The answer is [생각: 23+19=42] 42"

[생각]은 내부적으로만 사용
출력에는 포함 안 됨
```

### 2.2 학습 과정

```python
def quiet_star_forward(input_tokens):
    outputs = []

    for i, token in enumerate(input_tokens):
        # 1. 내부 생각 생성
        thought = generate_thought(input_tokens[:i])

        # 2. 생각을 조건으로 다음 토큰 예측
        hidden_with_thought = encode([*input_tokens[:i], thought])
        next_token = predict(hidden_with_thought)

        # 3. 생각 없이 예측
        hidden_without = encode(input_tokens[:i])
        next_token_base = predict(hidden_without)

        # 4. 어느 것이 더 좋은지 비교
        # → 생각이 도움되면 학습
```

### 2.3 REINFORCE 학습

```
보상 = 생각이 도움된 정도

loss = -log P(thought) × reward

생각이 예측을 개선하면 → 그 생각 강화
```

---

## 3. 핵심 기술

### 3.1 Mixing Head

```python
# 생각의 영향도를 동적으로 조절
mixing_weight = sigmoid(linear(hidden))

output = mixing_weight * with_thought + (1 - mixing_weight) * without_thought

→ 생각이 필요 없을 때는 0으로
→ 필요할 때만 활성화
```

### 3.2 Meta-Tokens

```
<|startofthought|>: 생각 시작
<|endofthought|>: 생각 끝

입력에 삽입하여 생각 구간 표시
```

---

## 4. 실험 결과

### 4.1 GSM8K

| 모델 | 정확도 |
|------|--------|
| Mistral 7B | 36.3% |
| + Quiet-STaR | **47.2%** |

+10.9% 향상 (프롬프팅 없이!)

### 4.2 CommonsenseQA

| 모델 | 정확도 |
|------|--------|
| Mistral 7B | 64.2% |
| + Quiet-STaR | **68.1%** |

### 4.3 특징

```
Zero-shot에서 개선:
- CoT 프롬프트 없이도
- 자동으로 추론 수행
```

---

## 5. 분석

### 5.1 어떤 토큰에서 생각하나?

```
분석 결과:
- 숫자 앞에서 많이 생각
- 결론 토큰 앞에서 많이 생각
- 단순 연결어에서는 적게

모델이 "언제 생각해야 하는지" 학습!
```

### 5.2 생각의 내용

```
예시:
입력: "If x + 3 = 7, then x ="
생각: "subtract 3 from both sides"
출력: "4"

실제로 추론하는 내용!
```

---

## 6. 핵심 요약

### 기억해야 할 것들

1. **핵심**: 모델이 내부적으로 생각하도록 학습
2. **방법**: 각 토큰 전에 thought 생성
3. **효과**: 프롬프팅 없이 추론 향상
4. **의의**: 내재된 추론 능력

### vs 기존 방법

| 방법 | 프롬프트 | 자동 |
|------|----------|------|
| CoT | 필요 | No |
| Self-Consistency | 필요 | No |
| **Quiet-STaR** | **불필요** | **Yes** |

### 향후 방향

```
- 더 큰 모델에 적용
- 다양한 태스크 확장
- 추론 시간 최적화
- o1/o3 등과의 관계
```

---

## 참고 자료

1. [Quiet-STaR 논문](https://arxiv.org/abs/2403.09629)

---

*이전 리뷰: [Tree of Thoughts](./003_Tree-of-Thoughts.md)*
*Part 4 완료!*
