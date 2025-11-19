# SimPO: Simple Preference Optimization with a Reference-Free Reward

**논문 발표**: 2024년
**저자**: Yu Meng, Mengzhou Xia, Danqi Chen
**소속**: Princeton University
**논문 링크**: [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)

---

## 한 줄 요약
> Reference 모델 없이 sequence의 평균 log probability를 reward로 사용하여, DPO보다 간단하면서 최대 6.4% 더 높은 성능 달성

---

## 1. DPO의 문제점 재고

### 1.1 Reference Model

```
DPO: log π_θ(y|x) - log π_ref(y|x)

문제:
- Reference 모델 메모리 필요
- 수학적으로 복잡
```

### 1.2 Length Bias

긴 응답이 낮은 log-prob → 불공정

---

## 2. SimPO

### 2.1 핵심 아이디어

1. Reference 제거: 직접 확률 사용
2. Length normalization: 평균 log-prob

### 2.2 SimPO Reward

$$r(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y|x)$$

### 2.3 SimPO Loss

$$\mathcal{L}_{\text{SimPO}} = -\log \sigma\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l|x) - \gamma\right)$$

- γ: Target reward margin

---

## 3. 실험 결과

### 3.1 AlpacaEval 2

| 모델 | DPO | **SimPO** |
|------|-----|-----------|
| Mistral-7B | 23.8 | **44.7** |
| LLaMA-3-8B | 33.9 | **40.5** |

### 3.2 MT-Bench

| 모델 | DPO | **SimPO** |
|------|-----|-----------|
| Mistral-7B | 7.32 | **7.56** |

---

## 4. 핵심 요약

### 기억해야 할 것들

1. **Reference-free**: π_ref 불필요
2. **Length normalized**: 평균 log-prob
3. **간단함**: 구현 매우 쉬움
4. **결과**: DPO 대비 최대 6.4% 향상

### SimPO vs DPO

| 측면 | DPO | SimPO |
|------|-----|-------|
| Reference | 필요 | **불필요** |
| Length norm | 없음 | **있음** |
| 메모리 | 많음 | **적음** |
| 성능 | 기준 | **+6.4%** |

---

## 참고 자료

1. [SimPO 논문](https://arxiv.org/abs/2405.14734)

---

*이전 리뷰: [SPIN](./006_SPIN.md)*
*다음 섹션: [Instruction Tuning](../3_Instruction_Tuning/)*
