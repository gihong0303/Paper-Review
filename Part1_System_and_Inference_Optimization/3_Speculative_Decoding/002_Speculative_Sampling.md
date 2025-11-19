# Accelerating Large Language Model Decoding with Speculative Sampling

**논문 발표**: 2023년
**저자**: Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper
**소속**: DeepMind
**논문 링크**: [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)

---

## 한 줄 요약
> Speculative Decoding과 동일한 시기에 독립적으로 개발된 방법으로, 수학적으로 동일한 rejection sampling 기반 접근법을 사용하여 2-2.5배 속도 향상 달성

---

## 1. 핵심 아이디어

### 1.1 Google의 Speculative Decoding과 동시 발견

두 논문이 거의 동시에 발표:
- **Leviathan et al. (Google)**: 2022년 11월
- **Chen et al. (DeepMind)**: 2023년 2월

수학적으로 **완전히 동일한** 접근법

### 1.2 핵심 차이점

| 측면 | Speculative Decoding | Speculative Sampling |
|------|---------------------|----------------------|
| 이론 전개 | 알고리즘 중심 | 수학적 증명 중심 |
| 실험 | 다양한 태스크 | Chinchilla 모델 |
| 기여 | 알고리즘 제안 | 이론적 분석 강화 |

---

## 2. 알고리즘 (동일)

### 2.1 Modified Rejection Sampling

```python
def speculative_sampling(target_p, draft_q, x):
    """
    target_p: Target 모델의 확률 분포
    draft_q: Draft 모델의 확률 분포
    x: Draft가 생성한 토큰
    """
    # 수용 확률
    accept_prob = min(1, target_p(x) / draft_q(x))

    if random() < accept_prob:
        return x, True  # 수용
    else:
        # 재샘플링
        adjusted_dist = normalize(max(0, target_p - draft_q))
        new_x = sample(adjusted_dist)
        return new_x, False  # 거부 및 수정
```

### 2.2 이론적 증명

**정리**: Speculative sampling의 출력 분포는 target 모델 p와 정확히 일치

**증명**:
$$P(X = x) = q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) + \left(1 - \sum_y q(y)\min\left(1, \frac{p(y)}{q(y)}\right)\right) \cdot \frac{\max(0, p(x) - q(x))}{\sum_z \max(0, p(z) - q(z))}$$

정리하면 $P(X = x) = p(x)$

---

## 3. 효율성 분석

### 3.1 기대 수용률

$$\alpha = \sum_x \min(p(x), q(x)) = 1 - \frac{1}{2}\|p - q\|_1$$

Draft와 Target의 분포가 유사할수록 높은 수용률

### 3.2 속도 향상 공식

$$\text{Speedup} = \frac{c \cdot \mathbb{E}[\text{accepted tokens}]}{1 + c \cdot (\gamma \cdot \text{draft cost})}$$

여기서:
- c: Target 대비 Draft 속도 비율
- γ: Lookahead 토큰 수

---

## 4. 실험 결과

### 4.1 Chinchilla 모델 (70B)

Draft 모델: 4B (17배 작음)

| 태스크 | 속도 향상 |
|--------|-----------|
| XSum | 2.01× |
| HumanEval | 2.46× |

### 4.2 수용률 분석

```
γ=4: 평균 3.2 토큰 수용 (80%)
γ=8: 평균 5.8 토큰 수용 (73%)
```

---

## 5. 주요 기여

### 5.1 이론적 기여

1. **분포 일치 증명**: 엄밀한 수학적 증명
2. **효율성 분석**: 속도 향상 공식 유도
3. **최적 γ 분석**: Lookahead 수 결정 방법

### 5.2 실무적 기여

- 대규모 모델(70B)에서 검증
- 코드 생성 태스크에서 효과 확인

---

## 6. 핵심 요약

### Speculative Decoding과 비교

두 논문은 **동일한 방법**을 독립적으로 발견:
- 수용/거부 기준 동일
- 재샘플링 방식 동일
- 분포 보장 동일

DeepMind 논문의 추가 기여:
- 더 엄밀한 이론적 분석
- 대규모 모델 실험

### 실무에서는?

```python
# 대부분의 구현은 두 논문을 동시에 인용
# 알고리즘적으로 동일하므로 구분 없이 사용
```

---

## 참고 자료

1. [Speculative Sampling 논문](https://arxiv.org/abs/2302.01318)
2. [DeepMind Blog](https://www.deepmind.com/)

---

*이전 리뷰: [Speculative Decoding](./001_Speculative_Decoding.md)*
*다음 리뷰: [Medusa](./003_Medusa.md)*
