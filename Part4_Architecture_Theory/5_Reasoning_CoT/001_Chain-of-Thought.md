# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**논문 발표**: 2022년 (NeurIPS 2022)
**저자**: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, et al.
**소속**: Google Research, Brain Team
**논문 링크**: [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)

---

## 한 줄 요약
> "단계별로 생각하자"라는 간단한 프롬프팅으로 LLM의 수학적 추론 능력을 획기적으로 향상

---

## 1. Chain-of-Thought (CoT)

### 1.1 기존 방식

```
Q: 카페에 23명이 있었습니다. 12명이 떠나고 6명이 왔습니다.
   지금 몇 명이 있나요?
A: 17

모델: ???
```

### 1.2 CoT 방식

```
Q: 카페에 23명이 있었습니다. 12명이 떠나고 6명이 왔습니다.
   지금 몇 명이 있나요?
A: 처음에 23명이 있었습니다.
   12명이 떠나서 23 - 12 = 11명이 남았습니다.
   6명이 와서 11 + 6 = 17명이 됩니다.
   답: 17

모델: 성공!
```

---

## 2. Few-shot CoT

### 2.1 프롬프트 예시

```python
prompt = """
Q: 사과가 5개 있습니다. 2개를 더 삽니다. 몇 개가 있나요?
A: 처음에 사과가 5개 있었습니다.
   2개를 더 사서 5 + 2 = 7개가 됩니다.
   답: 7

Q: 책이 15권 있습니다. 3권을 빌려주고 5권을 받았습니다.
   몇 권이 있나요?
A: 처음에 책이 15권 있었습니다.
   3권을 빌려줘서 15 - 3 = 12권이 남았습니다.
   5권을 받아서 12 + 5 = 17권이 됩니다.
   답: 17

Q: [새 문제]
A:
"""
```

### 2.2 Zero-shot CoT

```
"Let's think step by step."

이 한 문장만 추가해도 효과!
```

---

## 3. 실험 결과

### 3.1 GSM8K (수학)

| 방법 | PaLM 540B |
|------|-----------|
| Standard | 17.9% |
| **CoT** | **56.9%** |

3배 이상 향상!

### 3.2 모델 크기 효과

```
CoT는 큰 모델에서만 효과:

8B: CoT 거의 효과 없음
62B: 약간 효과
540B: 큰 효과

"Emergent ability"
```

---

## 4. 왜 효과적인가?

### 4.1 가설들

```
1. 문제 분해
   복잡한 문제 → 작은 단계들

2. 중간 결과 저장
   Working memory 역할

3. 오류 수정 기회
   각 단계에서 검증 가능
```

### 4.2 주의사항

```
CoT가 효과 없는 경우:
- 단순한 문제
- 작은 모델
- 사실 기반 질문 (추론 불필요)
```

---

## 5. 핵심 요약

### 기억해야 할 것들

1. **핵심**: 단계별 사고 과정 생성
2. **방법**: Few-shot 예시 또는 "Let's think step by step"
3. **효과**: 수학 추론 3배+ 향상
4. **조건**: 큰 모델에서만 효과적

### 적용 팁

```python
# Zero-shot CoT
prompt = f"{question}\n\nLet's think step by step."

# Few-shot CoT
prompt = f"{examples_with_reasoning}\n\n{question}\n\nAnswer:"
```

---

## 참고 자료

1. [CoT 논문](https://arxiv.org/abs/2201.11903)

---

*다음 리뷰: [Self-Consistency](./002_Self-Consistency.md)*
