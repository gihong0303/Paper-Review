# Self-Consistency Improves Chain of Thought Reasoning

**논문 발표**: 2022년 (ICLR 2023)
**저자**: Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, et al.
**소속**: Google Research
**논문 링크**: [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)

---

## 한 줄 요약
> 여러 추론 경로를 샘플링하고 다수결로 최종 답을 선택하여 CoT의 성능을 추가로 향상

---

## 1. 핵심 아이디어

### 1.1 문제

```
CoT도 틀릴 수 있음:
- 계산 실수
- 추론 오류
- 할루시네이션
```

### 1.2 해결: 다수결

```
같은 문제에 여러 번 추론:

시도 1: 답 = 17
시도 2: 답 = 15
시도 3: 답 = 17
시도 4: 답 = 17
시도 5: 답 = 15

다수결 → 답 = 17 (3표)
```

---

## 2. 알고리즘

```python
def self_consistency(question, model, n_samples=5):
    answers = []

    # 여러 번 샘플링
    for _ in range(n_samples):
        # Temperature > 0으로 다양한 응답
        response = model.generate(
            question,
            temperature=0.7,
            do_sample=True
        )
        # 최종 답 추출
        answer = extract_answer(response)
        answers.append(answer)

    # 다수결
    return majority_vote(answers)
```

---

## 3. 실험 결과

### 3.1 GSM8K

| 방법 | PaLM 540B |
|------|-----------|
| CoT (greedy) | 56.9% |
| **CoT + SC (40 paths)** | **74.4%** |

+17.5% 추가 향상!

### 3.2 샘플 수 효과

```
샘플 수 vs 성능:
1개: 56.9%
5개: 65.4%
10개: 69.1%
40개: 74.4%

더 많은 샘플 = 더 높은 성능
(수확 체감)
```

---

## 4. 왜 효과적인가?

### 4.1 오류 상쇄

```
각 추론에 무작위 오류:
- 일부는 틀린 답
- 다수는 맞는 답

다수결로 오류 상쇄!
```

### 4.2 다양성의 힘

```
Temperature로 다양성:
- 같은 추론만 반복 → 효과 없음
- 다양한 접근 → 효과 있음
```

---

## 5. 한계

### 5.1 비용

```
N번 추론 → N배 비용

40 샘플 = 40배 API 비용/시간
```

### 5.2 해결: Verifier

```
별도 모델로 답 검증:
- 답별로 점수
- 다수결 대신 점수 기반

더 효율적
```

---

## 6. 핵심 요약

### 기억해야 할 것들

1. **핵심**: 여러 번 추론 + 다수결
2. **효과**: CoT 대비 추가 15-20% 향상
3. **방법**: Temperature 샘플링
4. **비용**: 샘플 수에 비례

### 실무 팁

```python
# 권장 설정
n_samples = 5~10  # 비용 vs 성능 균형
temperature = 0.5~0.7  # 적절한 다양성
```

---

## 참고 자료

1. [Self-Consistency 논문](https://arxiv.org/abs/2203.11171)

---

*이전 리뷰: [Chain-of-Thought](./001_Chain-of-Thought.md)*
*다음 리뷰: [Tree of Thoughts](./003_Tree-of-Thoughts.md)*
