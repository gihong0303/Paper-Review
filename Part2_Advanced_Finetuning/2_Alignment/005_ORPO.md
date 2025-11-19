# ORPO: Monolithic Preference Optimization without Reference Model

**논문 발표**: 2024년
**저자**: Jiwoo Hong, Noah Lee, James Thorne
**소속**: KAIST
**논문 링크**: [arXiv:2403.07691](https://arxiv.org/abs/2403.07691)

---

## 한 줄 요약
> SFT와 preference alignment를 단일 목적함수로 통합하여, Reference 모델 없이 메모리와 연산을 절약하면서 DPO보다 우수한 성능 달성

---

## 1. DPO의 비효율

### 1.1 2단계 필요

```
기존:
1. SFT 학습 → Reference 모델
2. DPO 학습 → Reference 모델 필요

문제:
- 2번의 학습
- Reference 모델 메모리 (추가 GPU)
```

### 1.2 ORPO의 해결

```
ORPO: SFT + Alignment를 동시에
      Reference 모델 불필요!
```

---

## 2. ORPO Loss

### 2.1 수식

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{OR}}$$

$$\mathcal{L}_{\text{OR}} = -\log \sigma\left(\log \frac{p(y_w|x)}{1-p(y_w|x)} - \log \frac{p(y_l|x)}{1-p(y_l|x)}\right)$$

### 2.2 직관

- **SFT loss**: 좋은 응답 확률 높이기
- **OR loss**: 좋은/나쁜 응답의 odds ratio 최대화

---

## 3. 장점

1. **단일 단계**: SFT + Alignment 통합
2. **메모리 절약**: Reference 모델 불필요
3. **간단함**: 구현 용이

---

## 4. 실험 결과

| 모델 | 방법 | AlpacaEval | MT-Bench |
|------|------|------------|----------|
| Mistral-7B | DPO | 14.7 | 7.23 |
| Mistral-7B | **ORPO** | **17.6** | **7.32** |
| LLaMA-2-7B | DPO | 8.9 | 6.32 |
| LLaMA-2-7B | **ORPO** | **12.2** | **6.57** |

---

## 5. 핵심 요약

1. **핵심**: SFT + DPO 통합
2. **장점**: Reference 불필요, 메모리 절약
3. **결과**: DPO보다 우수

---

## 참고 자료

1. [ORPO 논문](https://arxiv.org/abs/2403.07691)

---

*이전 리뷰: [KTO](./004_KTO.md)*
*다음 리뷰: [SPIN](./006_SPIN.md)*
