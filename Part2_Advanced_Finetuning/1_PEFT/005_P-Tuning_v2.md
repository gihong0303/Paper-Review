# P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally

**논문 발표**: 2022년 (ACL 2022)
**저자**: Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, Jie Tang
**소속**: Tsinghua University
**논문 링크**: [arXiv:2110.07602](https://arxiv.org/abs/2110.07602)

---

## 한 줄 요약
> Prefix-Tuning을 개선하여 모든 레이어에 학습 가능한 prompt를 추가하고(Deep Prompt Tuning), 다양한 모델 크기와 태스크에서 Fine-tuning과 동등한 성능 달성

---

## 1. P-Tuning v1의 한계

- NLU 태스크에서 성능 저하
- 작은 모델에서 효과 없음
- Sequence labeling 태스크 미지원

---

## 2. P-Tuning v2의 개선

### 2.1 Deep Prompt Tuning

모든 레이어에 prompt 적용:

```
P-Tuning v1: 입력 레이어만
P-Tuning v2: 모든 레이어

Layer 1: [P1] + hidden
Layer 2: [P2] + hidden
...
Layer L: [PL] + hidden
```

### 2.2 추가 개선

- Reparameterization 제거 (직접 학습)
- 태스크별 최적화 기법
- Multi-task learning 지원

---

## 3. 실험 결과

| 모델 크기 | Fine-tune | P-Tuning v1 | **P-Tuning v2** |
|-----------|-----------|-------------|-----------------|
| 330M | 기준 | -5% | **동등** |
| 10B | 기준 | -2% | **동등** |

---

## 4. 핵심 요약

1. **핵심**: 모든 레이어에 prompt (Deep)
2. **장점**: 작은 모델에서도 효과적
3. **적용**: NLU, Sequence Labeling 등 다양한 태스크

---

## 참고 자료

1. [P-Tuning v2 논문](https://arxiv.org/abs/2110.07602)

---

*이전 리뷰: [Prefix-Tuning](./004_Prefix-Tuning.md)*
*다음 리뷰: [LISA](./006_LISA.md)*
