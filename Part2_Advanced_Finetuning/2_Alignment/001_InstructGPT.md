# InstructGPT: Training Language Models to Follow Instructions with Human Feedback

**논문 발표**: 2022년 (NeurIPS 2022)
**저자**: Long Ouyang, Jeff Wu, Xu Jiang, et al.
**소속**: OpenAI
**논문 링크**: [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

---

## 한 줄 요약
> RLHF(Reinforcement Learning from Human Feedback)를 통해 GPT-3를 인간의 의도에 맞게 정렬하여, 더 작은 모델(1.3B)이 더 큰 모델(175B)보다 선호되는 결과 달성 - ChatGPT의 전신

---

## 1. 문제: 언어 모델의 Misalignment

### 1.1 GPT-3의 문제점

- **Helpful하지 않음**: 지시를 잘 따르지 않음
- **Honest하지 않음**: 거짓 정보 생성
- **Harmless하지 않음**: 유해한 콘텐츠 생성

### 1.2 Alignment 목표

> "사용자가 원하는 것"에 맞추기

---

## 2. RLHF 3단계

### 2.1 전체 파이프라인

```
Step 1: SFT (Supervised Fine-Tuning)
        사람이 작성한 시범 데이터로 학습

Step 2: Reward Model Training
        사람의 선호도 비교로 보상 모델 학습

Step 3: PPO (Proximal Policy Optimization)
        보상 모델을 사용해 강화학습
```

### 2.2 Step 1: SFT

```python
# 사람이 직접 작성한 응답으로 학습
dataset = [
    {"prompt": "Write a poem about AI",
     "response": "In circuits deep and neural wide..."},
    ...
]
model.fine_tune(dataset)
```

### 2.3 Step 2: Reward Model

사람이 여러 응답을 순위 매김:

```
Prompt: "Explain quantum physics"
Response A: [상세한 설명] - 1위
Response B: [간단한 설명] - 2위
Response C: [틀린 설명]  - 3위
```

Reward model 학습:
$$\mathcal{L}_{RM} = -\log \sigma(r(x, y_w) - r(x, y_l))$$

### 2.4 Step 3: PPO

```python
# 강화학습으로 정책 최적화
objective = reward_model(response) - β * KL(policy || reference)
```

KL divergence로 원본 모델에서 너무 벗어나지 않게

---

## 3. 실험 결과

### 3.1 인간 평가

| 모델 | 선호도 |
|------|--------|
| GPT-3 175B | 기준 |
| **InstructGPT 1.3B** | **더 선호됨** |

**1.3B 모델이 175B보다 선호됨!**

### 3.2 품질 개선

- Truthfulness: +21%
- Toxicity: -25%
- Helpfulness: 크게 향상

---

## 4. RLHF의 한계

1. **비용**: 많은 인간 피드백 필요
2. **복잡성**: 3단계 파이프라인
3. **불안정**: PPO 학습 어려움
4. **Reward hacking**: 보상 모델 속이기

→ DPO 등 후속 연구가 이를 해결

---

## 5. 핵심 요약

### 기억해야 할 것들

1. **RLHF 3단계**: SFT → RM → PPO
2. **핵심**: 인간 선호도로 정렬
3. **결과**: 작은 모델이 큰 모델보다 나음
4. **의의**: ChatGPT의 기반

### 영향력

- ChatGPT, GPT-4의 기반
- Claude, Gemini 등 모든 현대 LLM에 적용
- Alignment 연구의 시작점

---

## 참고 자료

1. [InstructGPT 논문](https://arxiv.org/abs/2203.02155)
2. [OpenAI Blog](https://openai.com/blog/instruction-following/)

---

*다음 리뷰: [DPO](./002_DPO.md)*
