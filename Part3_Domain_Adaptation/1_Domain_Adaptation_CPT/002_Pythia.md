# Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling

**논문 발표**: 2023년 (ICML 2023)
**저자**: Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O'Brien, et al.
**소속**: EleutherAI
**논문 링크**: [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
**공식 구현**: [GitHub](https://github.com/EleutherAI/pythia)

---

## 한 줄 요약
> 70M부터 12B까지 8개 크기의 모델을 동일한 데이터로 학습하고 154개 체크포인트를 공개하여, 학습 역학과 스케일링을 체계적으로 분석할 수 있는 연구 플랫폼 제공

---

## 1. Pythia의 목적

### 1.1 기존 문제

- 학습 데이터 비공개
- 중간 체크포인트 없음
- 모델 간 비교 어려움

### 1.2 Pythia의 해결책

```
모든 것을 공개:
- 8개 모델 크기 (70M ~ 12B)
- 154개 체크포인트 (매 1000 steps)
- 학습 데이터 (The Pile)
- 학습 코드
```

---

## 2. 모델 구성

### 2.1 모델 크기

| 모델 | 파라미터 | 레이어 | Hidden | Heads |
|------|----------|--------|--------|-------|
| 70M | 70M | 6 | 512 | 8 |
| 160M | 160M | 12 | 768 | 12 |
| 410M | 410M | 24 | 1024 | 16 |
| 1B | 1B | 16 | 2048 | 8 |
| 1.4B | 1.4B | 24 | 2048 | 16 |
| 2.8B | 2.8B | 32 | 2560 | 32 |
| 6.9B | 6.9B | 32 | 4096 | 32 |
| 12B | 12B | 36 | 5120 | 40 |

### 2.2 학습 데이터

**The Pile** (300B tokens):
- 다양한 도메인 22개
- 고품질 영어 텍스트
- 중복 제거됨

---

## 3. 주요 분석 결과

### 3.1 데이터 순서의 영향

**실험**: 같은 데이터를 다른 순서로 학습

```
결과:
- 최종 성능: 거의 동일
- 중간 과정: 크게 다름
- 특정 능력 습득 시점: 순서에 따라 다름
```

### 3.2 중복 데이터의 영향

```
Standard Pythia: 중복 제거
Pythia-dedup: 더 철저한 중복 제거

결과:
- 중복 제거 → 약간 더 좋은 성능
- 특히 암기 감소
```

### 3.3 Gender Bias 학습 과정

```
학습 초기: Bias 낮음
학습 중기: Bias 급증
학습 후기: Bias 안정화

→ Bias는 학습 중 습득됨
```

---

## 4. Scaling Law 분석

### 4.1 Emergent Abilities

특정 능력이 특정 크기에서 "갑자기" 나타남:

```
예: Few-shot 능력
- 1B 이하: 거의 없음
- 2.8B: 나타나기 시작
- 6.9B+: 명확하게 존재
```

### 4.2 Task별 스케일링

```
태스크에 따라 스케일링 패턴이 다름:

일부: 로그 선형 개선
일부: 특정 크기에서 급격한 개선
일부: 크기와 무관
```

---

## 5. 한국어 연구 활용

### 5.1 체크포인트 활용

```python
from transformers import AutoModelForCausalLM

# 특정 체크포인트 로드
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-1.4b",
    revision="step100000"  # 중간 체크포인트
)
```

### 5.2 한국어 CPT 연구

Pythia 체크포인트로 실험:
1. 어느 시점부터 한국어 학습이 효과적인가?
2. 영어 능력은 얼마나 유지되는가?
3. 최적의 학습률은?

---

## 6. 핵심 요약

### 기억해야 할 것들

1. **공개**: 8개 크기, 154개 체크포인트
2. **발견**: 데이터 순서는 최종 성능에 영향 적음
3. **발견**: Bias는 학습 중 습득됨
4. **활용**: 스케일링 연구의 기초 자료

### 연구자를 위한 가치

- 학습 역학 연구 가능
- 재현 가능한 실험
- 비용 없이 대규모 모델 연구

---

## 참고 자료

1. [Pythia 논문](https://arxiv.org/abs/2304.01373)
2. [GitHub](https://github.com/EleutherAI/pythia)
3. [HuggingFace Models](https://huggingface.co/EleutherAI)

---

*이전 리뷰: [Don't Stop Pretraining](./001_Dont_Stop_Pretraining.md)*
*다음 리뷰: [ChipNeMo](./003_ChipNeMo.md)*
