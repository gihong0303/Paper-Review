# Fast Inference from Transformers via Speculative Decoding

**논문 발표**: 2022년 (ICML 2023)
**저자**: Yaniv Leviathan, Matan Kalman, Yossi Matias
**소속**: Google Research
**논문 링크**: [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

---

## 한 줄 요약
> 작은 Draft 모델로 여러 토큰을 빠르게 생성하고, 큰 Target 모델이 병렬로 검증하여 수용/거부하는 방식으로, 출력 분포를 정확히 유지하면서 2-3배 속도 향상

---

## 1. 핵심 아이디어

### 1.1 문제: Autoregressive 디코딩의 한계

```
토큰 1 생성 → 토큰 2 생성 → 토큰 3 생성 → ...
   │             │             │
   ▼             ▼             ▼
 모델 호출      모델 호출      모델 호출

각 토큰마다 전체 모델 실행 → 느림!
```

### 1.2 핵심 관찰

> "여러 토큰을 병렬로 검증하는 것은 순차적으로 생성하는 것만큼 비용이 들지 않는다"

Prefill처럼, 여러 토큰을 한 번에 처리 가능

### 1.3 Speculative Decoding 아이디어

```
Draft 모델 (작고 빠름):
  → 토큰 A, B, C, D 생성 (빠르게)

Target 모델 (크고 정확):
  → A, B, C, D를 병렬로 검증
  → A ✓, B ✓, C ✗ → A, B 채택, C 수정
```

---

## 2. 알고리즘

### 2.1 전체 과정

```
알고리즘: Speculative Decoding
─────────────────────────────────
입력: Target 모델 Mq, Draft 모델 Mp, 프롬프트

1. while not finished:
2.     # Draft 단계
3.     for i = 1 to γ:
4.         x_i ~ Mp(·|prefix)  # 빠른 생성
5.
6.     # Verification 단계
7.     q(x_1), ..., q(x_γ) = Mq(prefix, x_1:γ)  # 병렬!
8.
9.     # 수용/거부
10.    for i = 1 to γ:
11.        r ~ Uniform(0, 1)
12.        if r < min(1, q(x_i)/p(x_i)):
13.            accept x_i
14.        else:
15.            sample from (q - p)+ / sum  # 수정
16.            break
17.
18.    # 추가 토큰 (보너스)
19.    sample from q(·|accepted_tokens)
```

### 2.2 수용 확률

$$P(\text{accept } x) = \min\left(1, \frac{q(x)}{p(x)}\right)$$

- $q(x)$: Target 모델의 확률
- $p(x)$: Draft 모델의 확률

### 2.3 거부 시 재샘플링

```python
def resample(p, q):
    # Adjusted distribution
    diff = (q - p).clamp(min=0)
    diff = diff / diff.sum()
    return sample_from(diff)
```

**핵심**: 최종 출력이 Target 모델의 분포를 **정확히** 따름

---

## 3. 수학적 증명

### 3.1 분포 일치 증명

Speculative decoding의 출력 분포 = Target 모델의 출력 분포

**증명 개요**:
1. 수용 시: Draft의 x가 Target 분포와 일치하도록 수용
2. 거부 시: 남은 확률 질량에서 재샘플링

### 3.2 기대 수용 토큰 수

$$E[\text{# accepted}] = \sum_{i=1}^{\gamma} (1 - \alpha)^{i-1} \cdot \alpha$$

여기서 $\alpha = E_x[\min(1, q(x)/p(x))]$

Draft가 Target과 유사할수록 더 많이 수용

---

## 4. 쉬운 예시

### 4.1 시험 채점 비유

**기존 방식**:
- 교수님이 한 문제씩 채점 (느림)

**Speculative Decoding**:
- 조교가 먼저 5문제 채점 (빠름)
- 교수님이 5문제를 한 번에 검토
- 맞으면 채택, 틀리면 수정

### 4.2 코드 리뷰 비유

**기존**:
- 시니어 개발자가 한 줄씩 코딩

**Speculative**:
- 주니어가 먼저 여러 줄 작성
- 시니어가 병렬로 리뷰
- 맞으면 그대로, 틀리면 수정

---

## 5. 실험 결과

### 5.1 속도 향상

| 태스크 | 속도 향상 |
|--------|-----------|
| Translation | 2.5× |
| Summarization | 2.8× |
| Code Generation | 2.3× |

### 5.2 Lookahead 토큰 수 (γ)

| γ | 속도 | 수용률 |
|---|------|--------|
| 3 | 1.8× | 70% |
| 5 | 2.3× | 65% |
| 7 | 2.5× | 60% |

---

## 6. 구현 고려사항

### 6.1 Draft 모델 선택

```
요구사항:
- Target과 동일한 vocabulary
- 빠른 추론 (작은 크기)
- Target과 유사한 분포

예시:
- Target: LLaMA-70B
- Draft: LLaMA-7B 또는 TinyLLaMA
```

### 6.2 배치 처리

```python
def speculative_batch(target, draft, prompts, gamma=5):
    # 1. Draft 생성
    draft_tokens = draft.generate(prompts, n_tokens=gamma)

    # 2. Target 검증 (배치로)
    target_probs = target.forward(prompts + draft_tokens)

    # 3. 수용/거부 (병렬)
    accepted = verify_tokens(draft_tokens, target_probs)

    return accepted
```

---

## 7. 한계점

1. **Draft 모델 필요**: 별도 모델 로드/저장
2. **분포 불일치**: Draft가 Target과 다르면 효과 감소
3. **배치 비효율**: 각 시퀀스가 다른 수만큼 수용
4. **메모리**: 두 모델 동시 로드

---

## 8. 핵심 요약

### 기억해야 할 것들

1. **핵심**: Draft가 추측, Target이 검증
2. **장점**: 분포 정확히 유지하며 2-3× 속도 향상
3. **핵심 수식**: $P(\text{accept}) = \min(1, q/p)$
4. **한계**: Draft 모델 필요

### 코드 예시 (vLLM)

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_model="meta-llama/Llama-2-7b-hf",
    num_speculative_tokens=5
)

output = llm.generate("Hello")
```

---

## 참고 자료

1. [Speculative Decoding 논문](https://arxiv.org/abs/2211.17192)
2. [Google Research Blog](https://ai.googleblog.com/)

---

*이전 섹션: [Quantization](../2_Quantization/)*
*다음 리뷰: [Speculative Sampling (DeepMind)](./002_Speculative_Sampling.md)*
