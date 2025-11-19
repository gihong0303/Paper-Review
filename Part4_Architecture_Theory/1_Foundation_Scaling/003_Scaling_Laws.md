# Scaling Laws for Neural Language Models

**논문 발표**: 2020년
**저자**: Jared Kaplan, Sam McCandlish, Tom Henighan, et al.
**소속**: OpenAI, Johns Hopkins University
**논문 링크**: [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

---

## 한 줄 요약
> 언어 모델의 성능은 모델 크기, 데이터 양, 컴퓨팅의 멱법칙(power law)을 따르며, 이를 통해 최적의 자원 배분을 예측할 수 있음

---

## 1. 핵심 발견: 멱법칙

### 1.1 세 가지 스케일링 법칙

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

- $L$: Test Loss
- $N$: 파라미터 수
- $D$: 데이터 토큰 수
- $C$: 컴퓨팅 (FLOPs)

### 1.2 시각화

```
Loss vs Scale (log-log):

Loss
  │
  │ ╲
  │   ╲
  │     ╲
  │       ╲  직선!
  │         ╲
  └──────────────
              Scale

→ 멱법칙 = log-log에서 직선
```

---

## 2. 주요 결과

### 2.1 모델 크기 > 데이터 양

```
동일한 컴퓨팅 예산에서:
큰 모델 + 적은 데이터 > 작은 모델 + 많은 데이터

"모델을 키우는 게 더 효율적"
```

### 2.2 최적 배분

주어진 컴퓨팅 $C$에 대해:

$$N_{opt} \propto C^{0.73}$$
$$D_{opt} \propto C^{0.27}$$

```
컴퓨팅 10배 증가 시:
- 모델 크기: 5.4배 증가
- 데이터: 1.8배 증가
```

### 2.3 수렴 전에 멈춰도 OK

```
완전 수렴 vs 조기 종료:
- 완전 수렴: 컴퓨팅 낭비
- 조기 종료: 더 효율적

→ 큰 모델을 조기 종료하는 것이 최적
```

---

## 3. 아키텍처 독립성

### 3.1 놀라운 발견

```
Transformer 세부 설정과 무관:
- Depth vs Width
- Attention heads 수
- Feed-forward 크기

오직 총 파라미터 수만 중요!
```

### 3.2 단, 예외

```
중요한 것:
- 총 파라미터 N
- Context length (어느 정도)

무관한 것:
- Layer 수 vs Hidden size
- Head 수
```

---

## 4. 실무 적용

### 4.1 예산별 최적 설정

| 예산 (FLOPs) | 최적 N | 최적 D |
|--------------|--------|--------|
| 10^18 | 10M | 200M |
| 10^20 | 100M | 2B |
| 10^22 | 1B | 20B |
| 10^24 | 10B | 200B |

### 4.2 학습 계획

```python
def optimal_allocation(compute_budget):
    """주어진 컴퓨팅으로 최적 설정 계산"""
    N_opt = compute_budget ** 0.73 * constant_N
    D_opt = compute_budget ** 0.27 * constant_D
    return N_opt, D_opt

# 예: 10^23 FLOPs
N, D = optimal_allocation(1e23)
# N ≈ 5B, D ≈ 100B tokens
```

---

## 5. 한계와 후속 연구

### 5.1 이 논문의 한계

```
1. 학습 비용만 고려, 추론 비용 무시
2. 데이터 품질 무시
3. 특정 태스크 성능 예측 어려움
```

### 5.2 Chinchilla의 수정

```
Kaplan (2020): N을 키워라
Chinchilla (2022): N과 D를 균형있게!

→ 최적 비율이 다르게 계산됨
```

---

## 6. 쉬운 예시

### 6.1 투자 비유

```
자본 = 컴퓨팅
인력 = 모델 크기
데이터 = 경험

Scaling Laws:
"자본이 10배면, 인력 5배 + 경험 2배가 최적"

작은 팀을 오래 굴리는 것보다
큰 팀을 적당히 굴리는 게 효율적
```

---

## 7. 핵심 요약

### 기억해야 할 것들

1. **핵심**: 멱법칙 (Power Law)
2. **결론**: 모델을 키우는 게 효율적
3. **적용**: 컴퓨팅 예산으로 최적 N, D 계산
4. **주의**: Chinchilla가 이후 수정

### 핵심 수식

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}\right]^{\alpha_D}$$

---

## 참고 자료

1. [Scaling Laws 논문](https://arxiv.org/abs/2001.08361)

---

*이전 리뷰: [GPT-3](./002_GPT-3.md)*
*다음 리뷰: [Chinchilla](./004_Chinchilla.md)*
