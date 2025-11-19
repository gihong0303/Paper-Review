# Orca: Progressive Learning from Complex Explanation Traces of GPT-4

**논문 발표**: 2023년
**저자**: Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, Ahmed Awadallah
**소속**: Microsoft Research
**논문 링크**: [arXiv:2306.02707](https://arxiv.org/abs/2306.02707)

---

## 한 줄 요약
> 단순한 정답이 아닌 GPT-4의 상세한 사고 과정(explanation traces)을 학습하여, 작은 모델이 복잡한 추론 능력까지 획득

---

## 1. 기존 Distillation의 문제

### 1.1 Shallow Imitation

```
기존:
Q: "2+2는?"
A: "4"
→ 정답만 모방, 사고 과정 없음

Orca:
Q: "2+2는?"
A: "2+2를 계산하면, 2와 2를 더해서
    4가 됩니다. 따라서 답은 4입니다."
→ 사고 과정까지 학습
```

### 1.2 Explanation Traces

GPT-4에게 "step-by-step으로 설명하라"고 프롬프팅

---

## 2. Progressive Learning

### 2.1 단계별 학습

```
Stage 1: ChatGPT (쉬운 설명)에서 학습
         5M examples

Stage 2: GPT-4 (복잡한 설명)에서 학습
         1M examples
```

쉬운 것 → 어려운 것 순서로

---

## 3. 실험 결과

### 3.1 추론 벤치마크

| 모델 | BigBench Hard | AGIEval |
|------|---------------|---------|
| Vicuna-13B | 37.5 | 23.1 |
| **Orca-13B** | **47.5** | **32.1** |
| ChatGPT | 53.3 | 38.2 |

### 3.2 vs ChatGPT

복잡한 추론에서 ChatGPT의 95%+ 성능

---

## 4. 핵심 요약

1. **핵심**: 정답이 아닌 사고 과정 학습
2. **방법**: GPT-4의 explanation traces
3. **결과**: 추론 능력 크게 향상
4. **의의**: 효과적인 knowledge distillation

---

## 참고 자료

1. [Orca 논문](https://arxiv.org/abs/2306.02707)

---

*이전 리뷰: [LIMA](./001_LIMA.md)*
*다음 리뷰: [NEFTune](./003_NEFTune.md)*
