# NEFTune: Noisy Embeddings Improve Instruction Finetuning

**논문 발표**: 2023년 (ICLR 2024)
**저자**: Neel Jain, Ping-yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Aniruddha Saha, Micah Goldblum, Jonas Geiping, Tom Goldstein
**소속**: University of Maryland
**논문 링크**: [arXiv:2310.05914](https://arxiv.org/abs/2310.05914)

---

## 한 줄 요약
> 학습 시 임베딩에 무작위 노이즈를 추가하는 간단한 기법으로, 추가 데이터나 파라미터 없이 instruction following 성능을 최대 15% 향상

---

## 1. NEFTune 방법

### 1.1 핵심 아이디어

학습 시 입력 임베딩에 균일 노이즈 추가:

$$\tilde{e} = e + \frac{\alpha}{\sqrt{Ld}} \cdot U(-1, 1)$$

- $e$: 원본 임베딩
- $\alpha$: 노이즈 강도 (기본 5)
- $L$: 시퀀스 길이
- $d$: 임베딩 차원

### 1.2 구현

```python
def neftune_forward(embeddings, alpha=5):
    if training:
        L, d = embeddings.shape[-2:]
        noise = torch.zeros_like(embeddings).uniform_(-1, 1)
        embeddings = embeddings + alpha * noise / (L * d) ** 0.5
    return embeddings
```

---

## 2. 왜 효과적인가?

### 2.1 가설

1. **Regularization**: 과적합 방지
2. **Robustness**: 다양한 입력에 대한 일반화
3. **Exploration**: 학습 landscape 탐색

### 2.2 주의사항

추론 시에는 노이즈 제거!

---

## 3. 실험 결과

### 3.1 AlpacaEval

| 모델 | 기본 | +NEFTune | 개선 |
|------|------|----------|------|
| LLaMA-2-7B | 29.8 | **40.1** | +34% |
| LLaMA-2-13B | 34.1 | **45.2** | +33% |

### 3.2 Open LLM Leaderboard

평균 2-3% 향상

---

## 4. 핵심 요약

1. **방법**: 임베딩에 노이즈 추가
2. **코드 변경**: 1줄
3. **비용**: 없음
4. **효과**: 최대 15% 성능 향상

### 실무 적용

HuggingFace Trainer에서:
```python
trainer = Trainer(
    neftune_noise_alpha=5,  # NEFTune 활성화
    ...
)
```

---

## 참고 자료

1. [NEFTune 논문](https://arxiv.org/abs/2310.05914)

---

*이전 리뷰: [Orca](./002_Orca.md)*
*다음 섹션: [Distributed Training](../4_Distributed_Training/)*
