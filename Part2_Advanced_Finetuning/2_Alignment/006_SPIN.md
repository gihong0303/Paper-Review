# SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models

**논문 발표**: 2024년
**저자**: Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu
**소속**: UCLA
**논문 링크**: [arXiv:2401.01335](https://arxiv.org/abs/2401.01335)

---

## 한 줄 요약
> 모델이 자신의 응답과 실제 데이터를 구분하도록 반복 학습하여, 추가 인간 피드백 없이 지속적으로 성능을 개선하는 Self-play 방식

---

## 1. 핵심 아이디어

### 1.1 Self-Play 개념

```
Iteration 1:
  - 현재 모델이 응답 생성
  - 생성 응답 vs 실제 데이터 구분 학습

Iteration 2:
  - 개선된 모델이 더 좋은 응답 생성
  - 다시 구분 학습

반복 → 지속적 개선
```

### 1.2 GAN과 유사

Generator(모델)가 Discriminator(자기 자신)를 속이려 노력

---

## 2. SPIN Loss

$$\mathcal{L}_{\text{SPIN}} = \mathbb{E}\left[\ell\left(\lambda(r_\theta(y_{\text{real}}) - r_\theta(y_{\text{gen}}))\right)\right]$$

- 실제 데이터 확률 ↑
- 생성 데이터 확률 ↓

---

## 3. 실험 결과

| Iteration | MT-Bench |
|-----------|----------|
| 0 (SFT) | 5.94 |
| 1 | 6.78 |
| 2 | 6.95 |
| **3** | **7.04** |

반복할수록 성능 향상!

---

## 4. 핵심 요약

1. **핵심**: Self-play로 지속 개선
2. **장점**: 추가 인간 피드백 불필요
3. **방법**: 자신의 생성 vs 실제 데이터 구분

---

## 참고 자료

1. [SPIN 논문](https://arxiv.org/abs/2401.01335)

---

*이전 리뷰: [ORPO](./005_ORPO.md)*
*다음 리뷰: [SimPO](./007_SimPO.md)*
