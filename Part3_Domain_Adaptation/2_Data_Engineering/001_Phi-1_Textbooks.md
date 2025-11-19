# Textbooks Are All You Need: Phi-1

**논문 발표**: 2023년
**저자**: Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, et al.
**소속**: Microsoft Research
**논문 링크**: [arXiv:2306.11644](https://arxiv.org/abs/2306.11644)

---

## 한 줄 요약
> 양보다 질! "교과서 수준"의 고품질 합성 데이터 7B 토큰만으로 1.3B 파라미터 모델이 HumanEval 50%를 달성하며, 데이터 품질의 중요성을 입증

---

## 1. 핵심 아이디어

### 1.1 기존 접근법의 문제

```
기존 상식:
- 더 많은 데이터 = 더 좋은 모델
- 수조 개 토큰 필요

문제점:
- 웹 데이터의 낮은 품질
- 노이즈, 중복, 불완전한 코드
- 비효율적인 학습
```

### 1.2 Phi-1의 철학

```
"교과서처럼 배우자"

좋은 교과서의 특징:
- 명확한 설명
- 단계별 예제
- 핵심 개념 중심
- 일관된 스타일

→ 이런 데이터로 학습하면?
```

---

## 2. 데이터 구성

### 2.1 데이터셋 종류

| 데이터셋 | 토큰 수 | 설명 |
|----------|---------|------|
| Code Textbooks | 1B | 합성 교과서 |
| Code Exercises | 180M | 합성 연습문제 |
| Filtered Code | 6B | The Stack 필터링 |
| **총계** | **~7B** | - |

### 2.2 합성 데이터 생성

```python
# GPT-3.5로 교과서 스타일 데이터 생성
prompt = """
Write a textbook-style explanation of the following topic:
Topic: {topic}

Include:
- Clear explanation
- Step-by-step examples
- Common pitfalls
- Practice exercises
"""

topics = [
    "recursion in Python",
    "hash tables implementation",
    "graph traversal algorithms",
    ...
]

for topic in topics:
    textbook_content = gpt35.generate(prompt.format(topic=topic))
    dataset.add(textbook_content)
```

### 2.3 품질 필터링

```python
def filter_code(code_snippet):
    """The Stack에서 고품질 코드 필터링"""

    # 1. 교육적 가치 점수
    educational_score = gpt35.evaluate(
        f"Rate 0-10 how educational this code is: {code_snippet}"
    )

    # 2. 필터링
    if educational_score >= 7:
        return True
    return False
```

---

## 3. 모델 구조

### 3.1 Phi-1 스펙

| 항목 | 값 |
|------|-----|
| 파라미터 | 1.3B |
| 레이어 | 24 |
| Hidden | 2048 |
| Heads | 32 |
| Context | 2048 |

### 3.2 학습 설정

```python
training_config = {
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "warmup_steps": 750,
    "total_tokens": 50B,  # 7B 데이터를 여러 번 반복
    "optimizer": "AdamW",
    "weight_decay": 0.1
}
```

---

## 4. 왜 효과적인가?

### 4.1 고품질 데이터의 특징

```
1. 명확성 (Clarity)
   - 변수명이 의미 있음
   - 주석이 적절함
   - 로직이 단순명료

2. 완결성 (Completeness)
   - 전체 함수/클래스
   - import 문 포함
   - 에러 처리 포함

3. 교육성 (Educational)
   - 개념 설명 포함
   - 단계별 진행
   - 다양한 예제
```

### 4.2 대비: 일반 웹 데이터

```python
# 나쁜 예 (웹에서 흔함)
def f(x):
    return x*2+1  # ???

# 좋은 예 (교과서 스타일)
def double_and_increment(number: int) -> int:
    """
    주어진 숫자를 2배로 하고 1을 더합니다.

    Args:
        number: 입력 정수

    Returns:
        number * 2 + 1의 결과

    Example:
        >>> double_and_increment(5)
        11
    """
    doubled = number * 2
    result = doubled + 1
    return result
```

---

## 5. 실험 결과

### 5.1 HumanEval 성능

| 모델 | 파라미터 | 학습 토큰 | Pass@1 |
|------|----------|-----------|--------|
| CodeGen-Multi | 16B | 577B | 18.3% |
| StarCoder | 15B | 1T | 33.6% |
| **Phi-1** | **1.3B** | **7B** | **50.6%** |

**100배 적은 토큰으로 더 좋은 성능!**

### 5.2 효율성 비교

```
데이터 효율성:
Phi-1: 7B 토큰 → 50.6%
StarCoder: 1T 토큰 → 33.6%

Phi-1이 143배 더 효율적!
```

### 5.3 MBPP 결과

| 모델 | Pass@1 |
|------|--------|
| CodeGen-16B | 32.4% |
| PaLM-540B | 36.8% |
| **Phi-1-1.3B** | **44.8%** |

---

## 6. 추가 Fine-tuning: Phi-1.5를 향해

### 6.1 Code Exercises 효과

```
Phi-1-base: 29% (Textbooks만)
Phi-1: 50.6% (+ Exercises)

+21% 향상!
```

### 6.2 Exercise 데이터 예시

```python
# 연습 문제 형식
"""
Problem: Write a function that finds the longest common prefix
among a list of strings.

Example:
Input: ["flower", "flow", "flight"]
Output: "fl"
"""

def longest_common_prefix(strs):
    if not strs:
        return ""

    # 가장 짧은 문자열 찾기
    min_len = min(len(s) for s in strs)

    # 문자 비교
    for i in range(min_len):
        char = strs[0][i]
        if any(s[i] != char for s in strs):
            return strs[0][:i]

    return strs[0][:min_len]
```

---

## 7. 한계와 교훈

### 7.1 한계점

1. **범용성**: 코드 외 태스크는 약함
2. **언어**: Python 중심
3. **데이터 생성 비용**: GPT-3.5 API 비용

### 7.2 핵심 교훈

```
1. 데이터 양 < 데이터 질
2. 합성 데이터도 효과적
3. "교과서 스타일"이 학습에 최적
4. 작은 모델도 충분히 강력
```

---

## 8. 실무 적용

### 8.1 고품질 데이터 만들기

```python
# 교과서 스타일 데이터 생성 파이프라인
class TextbookDataGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate_explanation(self, topic):
        prompt = f"""
        Write a clear, educational explanation of {topic}.

        Requirements:
        - Start with motivation/why it matters
        - Explain core concepts step by step
        - Include 2-3 examples with increasing complexity
        - Mention common mistakes
        - End with a summary
        """
        return self.llm.generate(prompt)

    def generate_exercises(self, topic, difficulty="medium"):
        prompt = f"""
        Create coding exercises for {topic}.
        Difficulty: {difficulty}

        For each exercise:
        - Problem description
        - Input/Output examples
        - Solution with comments
        - Time/Space complexity
        """
        return self.llm.generate(prompt)
```

### 8.2 기존 데이터 필터링

```python
def filter_for_quality(dataset):
    """기존 데이터셋에서 교과서 수준만 선별"""
    filtered = []

    for item in dataset:
        # 품질 점수 (1-10)
        score = evaluate_educational_value(item)

        if score >= 8:
            filtered.append(item)

    return filtered  # 원본의 10-20%만 남음
```

---

## 9. 쉬운 예시

### 9.1 학습 비유

```
기존 방식 (양 중심):
- 도서관의 모든 책을 읽음
- 좋은 책, 나쁜 책 구분 없이
- 시간 많이 걸림, 혼란스러움

Phi-1 방식 (질 중심):
- 최고의 교과서만 선별
- 체계적으로 정리된 내용
- 빠르고 효과적인 학습
```

### 9.2 요리 학습 비유

```
기존: YouTube 요리 영상 1000개 시청
- 수준 들쭉날쭉
- 설명 부족한 영상 많음
- 시간 대비 효과 낮음

Phi-1: 전문 요리학교 교재 10권
- 기초부터 체계적
- 모든 과정 상세 설명
- 연습문제로 실력 확인
```

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **핵심 메시지**: 양보다 질!
2. **데이터**: 7B 토큰 (합성 교과서)
3. **결과**: 1.3B 모델로 50%+ HumanEval
4. **의의**: 데이터 품질이 모델 크기보다 중요

### 주요 수치

| 항목 | Phi-1 |
|------|-------|
| 파라미터 | 1.3B |
| 학습 토큰 | 7B |
| HumanEval | 50.6% |
| 효율성 | 143x |

### 수식으로 표현

$$\text{Model Quality} = f(\text{Data Quality}) > g(\text{Data Quantity})$$

### 실무 팁

- GPT-3.5/4로 합성 데이터 생성
- 기존 데이터는 품질 필터링
- "교과서처럼" 작성
- 연습문제 포함이 중요

---

## 참고 자료

1. [Phi-1 논문](https://arxiv.org/abs/2306.11644)
2. [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/)

---

*다음 리뷰: [Phi-1.5](./002_Phi-1.5.md)*
