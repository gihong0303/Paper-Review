# Tree of Thoughts: Deliberate Problem Solving with LLMs

**논문 발표**: 2023년 (NeurIPS 2023)
**저자**: Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, et al.
**소속**: Princeton University, Google DeepMind
**논문 링크**: [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)

---

## 한 줄 요약
> BFS/DFS 탐색 알고리즘을 프롬프팅에 적용하여 여러 추론 경로를 체계적으로 탐색하고 평가

---

## 1. CoT의 한계

### 1.1 문제

```
CoT = 단일 선형 경로

한 번 잘못된 방향 → 끝까지 잘못됨
되돌아가기 불가능
```

### 1.2 ToT 해결

```
Tree of Thoughts:
- 여러 경로 동시 탐색
- 각 단계 평가
- 좋은 경로만 확장

트리 탐색 알고리즘!
```

---

## 2. 알고리즘

### 2.1 구조

```
                 [문제]
                    │
        ┌───────────┼───────────┐
     [생각1]     [생각2]     [생각3]
        │           │           │
     ┌──┴──┐     ┌──┴──┐       ✗
    [1a]  [1b]  [2a]  [2b]
     │     ✗     │     │
    답1         답2   답3
```

### 2.2 구성 요소

```python
def tree_of_thoughts(problem):
    # 1. 생각 생성 (Thought Generator)
    thoughts = generate_thoughts(problem, n=3)

    # 2. 상태 평가 (State Evaluator)
    scores = evaluate_thoughts(thoughts)

    # 3. 탐색 알고리즘 (Search Algorithm)
    for thought in bfs_or_dfs(thoughts, scores):
        if is_solution(thought):
            return thought
        # 다음 단계 확장
        next_thoughts = generate_thoughts(thought)
        ...
```

---

## 3. 핵심 구성요소

### 3.1 Thought Generator

```python
def generate_thoughts(state, n=5):
    """현재 상태에서 가능한 다음 생각들 생성"""
    prompt = f"""
    Current state: {state}

    Generate {n} different next steps:
    """
    return model.generate(prompt, n=n)
```

### 3.2 State Evaluator

```python
def evaluate_thought(state, thought):
    """생각의 유망함 평가"""
    prompt = f"""
    State: {state}
    Thought: {thought}

    Rate this thought (1-10):
    Is it promising for solving the problem?
    """
    return model.generate(prompt)
```

### 3.3 Search Algorithm

```python
# BFS (넓은 탐색)
from queue import Queue
q = Queue()
q.put(initial_state)

while not q.empty():
    state = q.get()
    thoughts = generate_thoughts(state)
    for t in filter_best(thoughts):
        q.put(t)

# DFS (깊은 탐색)
def dfs(state, depth):
    if is_solution(state):
        return state
    thoughts = generate_thoughts(state)
    for t in thoughts:
        result = dfs(t, depth+1)
        if result:
            return result
    return None
```

---

## 4. 실험 결과

### 4.1 Game of 24

```
4개 숫자로 24 만들기:
예: 4, 9, 10, 13 → (10 - 4) × (13 - 9) = 24

CoT: 7.3%
ToT: 74%

10배 향상!
```

### 4.2 Creative Writing

```
4문단 이야기 쓰기:
각 문단이 특정 조건 만족

CoT: 낮은 일관성
ToT: 높은 일관성
```

---

## 5. 장단점

### 5.1 장점

```
1. 되돌아가기 가능
2. 여러 경로 비교
3. 체계적 탐색
4. 복잡한 문제에 강함
```

### 5.2 단점

```
1. 많은 API 호출
2. 느림
3. 단순 문제에는 과도함
```

---

## 6. 핵심 요약

### 기억해야 할 것들

1. **핵심**: 트리 탐색으로 추론
2. **구성**: 생성 + 평가 + 탐색
3. **효과**: 복잡한 추론 문제 해결
4. **비용**: 많은 API 호출 필요

### 적용 시점

```
ToT가 필요한 경우:
- 여러 단계 계획
- 조합 문제
- 창의적 과제

불필요한 경우:
- 단순 QA
- 계산만 필요
```

---

## 참고 자료

1. [Tree of Thoughts 논문](https://arxiv.org/abs/2305.10601)

---

*이전 리뷰: [Self-Consistency](./002_Self-Consistency.md)*
*다음 리뷰: [Quiet-STaR](./004_Quiet-STaR.md)*
